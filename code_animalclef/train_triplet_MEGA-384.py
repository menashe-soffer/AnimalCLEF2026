import matplotlib
from kornia.metrics import accuracy
from sklearn.metrics import accuracy_score

#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
from tqdm import tqdm
import torch.nn.functional as F
import copy
import gc
import numpy as np
import sklearn.metrics
from pytorch_metric_learning import miners, losses, samplers

from wildlife_datasets.datasets import AnimalCLEF2026
from AnimalCLEF_triplet_dataset import AnimalCLEFTripletDataset

from paths_and_constants import *
from my_models import AnimalReIDRefiner
from image_tools import UnderwaterEnhance, AddGaussianNoise, TextureHarvest
from model_featue_config import model_feature_config
from PKsampler import PKSampler, AbsoluteConstraintLoss




def train_model(model, dataset, output_fname, epochs=10, lr=1e-4, batch_size=8+4, weighted_data=False, device='cuda', as_classifier=False):

    model.to(device)

    # Only optimize parameters that are NOT frozen (Stages 3/4 + Head)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-3)
    #
    #

    # Standard Triplet Margin Loss
    #criterion = torch.nn.TripletMarginLoss(margin=0.5, p=2)
    if as_classifier:
        criterion = torch.nn.CrossEntropyLoss()
        output_fname = copy.copy(output_fname).replace('.pth', '.cls.pth')
    else:
        criterion = AbsoluteConstraintLoss(pos_margin=0.2, neg_margin=0.6)


    if weighted_data:
        dataset.use_split('trn')
        sample_freqs = dataset.get_sample_frequency()
        sample_weights = 1.0 / np.sqrt(np.maximum(3, sample_freqs))


    best_avg_val = 999
    best_sil_score = -1
    trn_loss_trc, val_loss_trc, sil_score_trc, mines_triplets_count_trc, accuracy_score_trc = [], [], [], [], []

    # miner = miners.TripletMarginMiner(margin=0.25, type_of_triplets="hard")
    # trn_criterion = losses.TripletMarginLoss(margin=0.25)
    # #trn_criterion = losses.ContrastiveLoss(pos_margin=0.2, neg_margin=0.5)

    for epoch in range(epochs):
        # --- TRAINING PHASE ---

        all_embeddings_list, all_labels_list = [], []  # for calculating val silhouette_score

        # if epoch > 25:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.98
        # boost lr
        if epoch == 0:
            optimizer.param_groups[0]['lr'] = train_cfg['lr'] * 3
        if epoch == 25:
            optimizer.param_groups[0]['lr'] = train_cfg['lr']

        dataset.use_split('trn')

        pk_sampling = False # TBD move to config
        if pk_sampling:
            anchor_idxs, anchor_ids = dataset.get_anchor_idxs_and_ids()
            #sampler = PKSampler(anchors=anchor_idxs, labels=anchor_ids, p=2, k=4)
            sampler = samplers.MPerClassSampler(dataset.get_anchor_idxs_and_ids()[1], m=4, batch_size=batch_size)
            loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=2)
        if (not pk_sampling) and weighted_data:
            sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=dataset.__len__(), replacement=True)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4,pin_memory=True, drop_last=True)
        if (not pk_sampling) and (not weighted_data):
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        num_triplets = 0

        model.train()
        running_trn_loss = 0.0

        num_steps = int(dataset.__len__() / batch_size)
        step = 0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} [TRN]")
        for batch in pbar:
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)

            # Forward passes
            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)

            # # semi-hard mining
            # anc_labels = batch['anchor_id']#torch.tensor(batch['anchor_id'])
            # neg_labels = batch['neg_id']#torch.tensor(batch['neg_id'])
            # all_embeddings = torch.cat([a_emb, p_emb, n_emb], dim=0)
            # all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            # combined_labels = torch.cat([anc_labels, anc_labels, neg_labels], dim=0)
            # indices_tuple = miner(all_embeddings, combined_labels)
            # #
            # num_triplets += indices_tuple[0].size(0)
            # #print(f"Batch {0}: Miner found {num_triplets} triplets")
            # #

            # # collect data for silhouette_score
            # all_embeddings_list.append(torch.cat((a_emb, p_emb, n_emb), dim=0).detach().cpu().numpy())
            # all_labels_list.append(np.concatenate((batch['anchor_id'], batch['anchor_id'], batch['neg_id'])))

            #loss = criterion(a_emb, p_emb, n_emb)
            #loss = trn_criterion(all_embeddings, combined_labels, indices_tuple)
            #print(torch.Tensor(batch['anchor_id']).long().cuda())
            if as_classifier:
                anchor_labels = batch['anchor_id'].long().cuda()
                neg_labels = batch['neg_id'].long().cuda()

                loss = (criterion(a_emb, anchor_labels) +
                        criterion(p_emb, anchor_labels) +
                        criterion(n_emb, neg_labels)) / 3
            else:
                loss = criterion(a_emb, p_emb, n_emb)

            loss.backward()

            step += 1
            if (step % 2 == 0) or (step == num_steps):#True:#
                optimizer.step()
                optimizer.zero_grad()

            running_trn_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_trn = running_trn_loss / len(loader)

        # --- VALIDATION PHASE ---
        dataset.use_split('val')
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [VAL]")
            for batch in pbar:
                a_emb = model(batch['anchor'].to(device))
                p_emb = model(batch['positive'].to(device))
                n_emb = model(batch['negative'].to(device))

                if as_classifier:
                    anchor_labels = batch['anchor_id'].long().cuda()
                    neg_labels = batch['neg_id'].long().cuda()

                    loss = (criterion(a_emb, anchor_labels) +
                            criterion(p_emb, anchor_labels) +
                            criterion(n_emb, neg_labels)) / 3
                else:
                    loss = criterion(a_emb, p_emb, n_emb)

                running_val_loss += loss.item()

                # collect data for silhouette_score
                all_embeddings_list.append(torch.cat((a_emb, p_emb, n_emb), dim=0).detach().cpu().numpy())
                all_labels_list.append(np.concatenate((batch['anchor_id'], batch['anchor_id'], batch['neg_id'])))

        avg_val = running_val_loss / len(val_loader)

        if epoch == 0:
            sil_score = -1
            accuracy = 0
        else:
            if (not as_classifier) and (epoch % 10 == 0):
                sil_score = sklearn.metrics.silhouette_score(np.concatenate(all_embeddings_list, axis=0), np.concatenate(all_labels_list, axis=0), metric='cosine')
        if as_classifier:
            detect = np.argmax(all_embeddings_list[0], axis=1).flatten()
            cm = sklearn.metrics.confusion_matrix(all_labels_list[0], detect)
            accuracy = np.diag(cm).sum() / cm.sum()

        if as_classifier:
            print(f"Epoch {epoch + 1} Complete | TRN Loss: {avg_trn:.4f} | VAL Loss: {avg_val:.4f} | accuracy score: {accuracy:.4f}")
        else:
            print(f"Epoch {epoch + 1} Complete | TRN Loss: {avg_trn:.4f} | VAL Loss: {avg_val:.4f} | silhouette score: {sil_score:.4f}")
            print('avg num (mined) triplets per anchor:', num_triplets / (num_steps * batch_size))
        trn_loss_trc.append(avg_trn)
        val_loss_trc.append(avg_val)
        sil_score_trc.append(sil_score)
        accuracy_score_trc.append(accuracy)
        mines_triplets_count_trc.append(num_triplets / (num_steps * batch_size))

        if avg_val < best_avg_val:
            # Save the specific weights
            best_avg_val = avg_val
            torch.save(model.state_dict(), output_fname)
            print('saved', output_fname)
        if sil_score > best_sil_score:
            # Save the specific weights
            best_sil_score = sil_score
            output_fname_sil = output_fname.replace('.pth', '_sil.pth')
            torch.save(model.state_dict(), output_fname_sil)
            print('saved', output_fname_sil)

        if epoch > 2:
            try:
                plt.clf()
            except:
                pass
        if epoch >= 2:
            plt.plot(trn_loss_trc, label='TRN')
            plt.plot(val_loss_trc, label='VAL')
            #plt.plot((np.array(sil_score_trc) + 1) / 2, label='sil')
            if as_classifier:
                plt.plot(np.array(accuracy_score_trc), label='accuracy')
            if not as_classifier:
                plt.plot((1 - np.array(sil_score_trc)) / 4, label='(1-sil)/4')
            plt.ylim((0, min(0.6+1.4*as_classifier, 1.1 * max(np.max(trn_loss_trc), np.max(val_loss_trc), np.max(sil_score_trc)))))
            plt.grid(True)
            plt.legend()
            # plt.show(block=False)
            # plt.pause(0.1)
            plt.savefig(os.path.join(ROOT_MODELS, output_fname.replace('.pth', '_convergence.png')))
            plt.close()

        # maintain vram
        gc.collect()
        torch.cuda.empty_cache()


    return model




def train_main(dset_name, cfg, num_epochs=5, ena_singlton=False, weighted_data=True, as_classifier=False):


    # (1) Generate and configure models
    dataset_full = AnimalCLEF2026(
        ROOT_DATA,
        transform=None,
        load_label=True,  # return label as 2nd parameter
        factorize_label=True,  # replace string for unique integer
        check_files=False
    )
    base_dataset = dataset_full.get_subset(dataset_full.df['dataset'] == dset_name)
    triplet_ds = AnimalCLEFTripletDataset()
    triplet_ds.enable_singletons(trn_enabled=ena_singlton, val_enabled=True)
    big_id_list = [1, 26, 5, 8, 10, 23, 21, 14, 27, 40, 9, 25, 4, 20, 16, 6]
    triplet_ds.attach_dataset(base_dataset=base_dataset, max_allowed_class_size=None,
                              exclude_ID_list=[] if as_classifier else big_id_list ,
                              merge_IDs_not_in=big_id_list if as_classifier else [],
                              include_test=False)

    # We use your generic class from before
    projection_dim = len(big_id_list) + 1 if as_classifier else 256
    model = AnimalReIDRefiner(model_name=cfg['model_name'], weights_file=None, use_projector=cfg['use_projector'], projection_dim=projection_dim)
    #model.freeze_for_training(active_stages=[2, 3])

    # (2) define transforms
    # Since you're working with 384x384 (Mega-384), we use that size.
    train_transforms = T.Compose([
        #UnderwaterEnhance,  # First, fix the visibility
        T.Resize(cfg['size']),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomGrayscale(0.2),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        AddGaussianNoise(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if cfg['enhance']:
        train_transforms = T.Compose([UnderwaterEnhance, train_transforms])

    val_transforms = T.Compose([
        #UnderwaterEnhance,  # First, fix the visibility
        T.Resize(cfg['size']),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if cfg['enhance']:
        val_transforms = T.Compose([UnderwaterEnhance, val_transforms])

    # Configure transforms (using the ones we discussed earlier)
    triplet_ds.config_trn_transforms(train_transforms)
    triplet_ds.config_val_transforms(val_transforms)

    # (4) Refine Specifically for Turtles
    # You might want to filter the base_ds to turtles ONLY before this step
    print("\n--- Starting {} Refinement ---".format(dset_name))
    output_fname = os.path.join(ROOT_MODELS,'{}.pth'.format(cfg['wgt_file']))
    refined_model = train_model(model=model, dataset=triplet_ds, output_fname=output_fname, epochs=num_epochs, weighted_data=weighted_data, lr=cfg['lr'], as_classifier=as_classifier)

    # # Save the specific weights
    output_fname_final = output_fname.replace('.pth', '_final.pth')
    torch.save(refined_model.state_dict(), output_fname_final)
    print('saved', output_fname_final)


if __name__ == '__main__':

    dset_name = 'LynxID2025'

    config = model_feature_config()
    config.select_config_version('rsrch')
    train_cfg = config.get_training_config(dset_name)
    train_main(dset_name, train_cfg, ena_singlton=True, weighted_data=False, num_epochs=25+75, as_classifier=True) # stage 1 - classification (of big IDs)
    train_main(dset_name, train_cfg, ena_singlton=True, weighted_data=False, num_epochs=25+75, as_classifier=False) # stage 2 - clusstering (of remaining)

    plt.show() # hold the plot

