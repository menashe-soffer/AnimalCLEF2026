import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T
from tqdm import tqdm
import torch.nn.functional as F
import os
import gc
import time
from wildlife_datasets.datasets import AnimalCLEF2026

from paths_and_constants import *
from AnimalCLEF_contrastive_dataset import AnimalCLEFContrastiveDataset  # Your new class
from image_tools import UnderwaterEnhance
from my_models import AnimalReIDRefiner
from monitoring import print_vram_stats
from my_metrics import calculate_mrr_numpy

class SupConLoss(torch.nn.Module):
    """Handles both NTXent (unlabeled) and SupCon (labeled) based on labels."""

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temp = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]  # This is 2N

        # 1. L2 Normalize embeddings
        features = F.normalize(features, dim=1)

        # 2. Similarity Matrix (2N x 2N)
        logits = torch.matmul(features, features.T) / self.temp

        # 3. Create the Mask (Positives have same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 4. Remove self-similarity from mask
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # 5. Compute Log-Softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # 6. Mean Log-Likelihood
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        return -mean_log_prob_pos.mean()



class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z, labels):#z_i, z_j):
        """
        z_i: [batch_size, dim] - Embeddings from first augmentation
        z_j: [batch_size, dim] - Embeddings from second augmentation
        """
        # batch_size = z_i.shape[0]
        #
        # # 1. Concatenate and Normalize
        # z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, dim]
        # z = F.normalize(z, dim=1)

        # labels is ignored

        batch_size = int(z.shape[0] / 2)

        # 2. Compute Similarity Matrix (2nd view compares to 1st and vice versa)
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # [2N, 2N]

        # 3. Create Mask to remove self-similarity (the diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # 4. Create Targets
        # For image i, the positive is at i + batch_size (and vice versa)
        targets = torch.arange(2 * batch_size).to(z.device)
        targets[:batch_size] += batch_size
        targets[batch_size:] -= batch_size

        # 5. Cross Entropy treats this as a (2N-1) classification problem
        loss = F.cross_entropy(sim_matrix, targets)
        return loss





def train_step(model, loader, device, optimizer, criterion, pbar_str, collect_data=False):

    running_loss = 0.0
    # print_vram_stats()

    data_collection = None
    aggr_cycle, aggr_counter = 2, 1
    pbar = tqdm(loader, desc=pbar_str)
    for batch in pbar:
        # batch contains 'im_q', 'im_k', and 'label'
        im_q = batch['im_q'].to(device)
        im_k = batch['im_k'].to(device)
        labels = batch['label'].to(device)

        # Two separate forward passes = half the peak memory
        features_q = model(im_q)
        features_k = model(im_k)

        # Re-stack them ONLY for the loss function (very cheap)
        features = torch.cat([features_q, features_k], dim=0)
        features = F.normalize(features, dim=1)
        multi_labels = torch.cat([labels, labels], dim=0)
        if optimizer is not None:
            optimizer.zero_grad()

        if collect_data:
            if data_collection is None:
                data_collection = dict({'features': features.detach().cpu(), 'labels': multi_labels.detach().cpu()})
            else:
                data_collection['features'] = torch.cat([data_collection['features'], features.detach().cpu()], dim=0)
                data_collection['labels'] = torch.cat([data_collection['labels'], multi_labels.detach().cpu()], dim=0)

        loss = criterion(features, multi_labels)
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader)

    # print_vram_stats()
    # del im_q, im_k, features, loss
    # torch.cuda.empty_cache()
    # gc.collect()
    # print_vram_stats()


    return  avg_loss, data_collection


def train_contrastive(model, dataset, output_fname, epochs=5, lr=1e-4, batch_size=8, loss_type='SupCon', temperature=0.07, device='cuda'):

    model.to(device)

    if loss_type == 'SupCon':
        criterion = SupConLoss(temperature=temperature)
        proj_lr = 4 * lr
        wgt_decay = 20 * lr
    if loss_type == 'NTXent':
        criterion = NTXentLoss(temperature=temperature)
        proj_lr = 10 * lr
        wgt_decay = 1 * lr

    backbone_params = model.backbone.parameters()  # Or your specific layers
    projector_params = model.projector.parameters()
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': proj_lr},  # Stable fine-tuning
        {'params': projector_params, 'lr': proj_lr}  # Aggressive learning for the new head
    ], weight_decay=wgt_decay)

    best_loss = 999
    best_mmr = -1
    trc_trn_loss, trc_val_loss, trc_val_mmr = [], [], []

    for epoch in range(epochs):

        # train step
        model.train()
        dataset.set_trn()
        id_frequencies = dataset.df['assigned_label'].value_counts().to_dict()
        weights = [1.0 / np.sqrt(max(id_frequencies[label], 3)) for label in dataset.df['assigned_label']]
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)          # Crucial for oversampling rare IDs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

        avg_trn_loss, _ = train_step(model, loader, device, optimizer, criterion,
                                     pbar_str=f"Epoch {epoch + 1}/{epochs} [{loss_type}]")

        # val step
        model.eval()
        dataset.set_val()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        avg_val_loss, val_data = train_step(model, loader, device, None, criterion,
                                            pbar_str=f"Epoch {epoch + 1}/{epochs} [{loss_type}] (VAL)", collect_data=True)

        genuine_mask = np.array([dataset.is_genuine(l) for l in val_data['labels']], dtype=bool)
        enable_mmr = genuine_mask.sum() > genuine_mask.size / 2
        if enable_mmr:
            features_np = val_data['features'].cpu().numpy()[genuine_mask]
            labels_np = val_data['labels'].cpu().numpy()[genuine_mask]
            mmr = calculate_mrr_numpy(features=features_np, labels=labels_np)
            print(f"Epoch {epoch + 1} Complete | Avg Loss: TRN: {avg_trn_loss:.4f}  VAL: {avg_val_loss:.4f}  MMR: {mmr:.4f}")
        else:
            print(f"Epoch {epoch + 1} Complete | Avg Loss: TRN: {avg_trn_loss:.4f}  VAL: {avg_val_loss:.4f}")
        #
        #print(f"Epoch {epoch + 1} Complete | Avg Loss: TRN: {avg_trn_loss:.4f}  VAL: {avg_val_loss:.4f}  MMR: {mmr:.4f}")
        trc_trn_loss.append(avg_trn_loss)
        trc_val_loss.append(avg_val_loss)
        if enable_mmr:
            trc_val_mmr.append(mmr)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save state_dict (we'll remember to strip 'model.' later if needed)
            torch.save(model.state_dict(), output_fname)
            print(f"saving weights to {output_fname}")
        if enable_mmr and (mmr > best_mmr):
            best_mmr = mmr
            output_fname_mmr = output_fname.replace('.pth', '_mmr.pth')
            torch.save(model.state_dict(), output_fname_mmr)
            print(f"saving weights to {output_fname_mmr}")

    return model, trc_trn_loss, trc_val_loss, trc_val_mmr


# for Labeled SupCon (Turtles)
weak_train_transforms = T.Compose([
    UnderwaterEnhance,
    T.RandomResizedCrop(384, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

lynx_trn_transform = T.Compose([
    # No UnderwaterEnhance here!
    T.RandomResizedCrop(384, scale=(0.8-0.25, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1), # Keep it subtle
    #T.ColorJitter(brightness=0.25, contrast=0.2, hue=0.25, saturation=0.25), # Keep it subtle
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

lynx_val_transform = T.Compose([
    # No UnderwaterEnhance here!
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


TRN_PARAMS = dict()
TRN_PARAMS['SalamanderID2025'] = dict({'epochs': 20, 'lr': 1e-4, 'transform': None, 'loss_type': 'NTXent', 'temperature': 0.12, 'val_ratio': 0.2})
TRN_PARAMS['SeaTurtleID2022'] = dict({'epochs': 8, 'lr': 1e-4, 'transform': weak_train_transforms, 'loss_type': 'SupCon', 'temperature': 0.07, 'val_ratio': 0.2})
TRN_PARAMS['LynxID2025'] = dict({'epochs': 8, 'lr': 1e-4, 'transform': lynx_trn_transform, 'loss_type': 'SupCon', 'temperature': 0.05, 'val_ratio': 0.2})
TRN_PARAMS['TexasHornedLizards'] = dict({'epochs': 20, 'lr': 1e-4, 'transform': None, 'loss_type': 'NTXent', 'temperature': 0.15, 'val_ratio': 0.2})



def main(db_name):

    trn_params = TRN_PARAMS[db_name]
    print('\n\ntraining with dataset', db_name)

    loss_type = trn_params['loss_type']   # 'SupCon', 'NTXent'

    # 1. Setup Base Data
    dataset_full = AnimalCLEF2026(root=ROOT_DATA)
    # This subset can now include your 'test' split for lizards!
    base_dataset = dataset_full.get_subset(dataset_full.df['dataset'] == db_name)

    # 2. Setup Contrastive Dataset
    contrastive_ds = AnimalCLEFContrastiveDataset()
    contrastive_ds.attach_dataset(base_dataset)
    contrastive_ds.make_split(val_ratio=trn_params['val_ratio'])

    contrastive_ds.config_transforms(transforms_trn=trn_params['transform']) # set default for now

    # 4. Model and Training
    model = AnimalReIDRefiner(use_projector=True)
    model.freeze_for_training(active_stages=[3])

    output_path = os.path.join(ROOT_MODELS,'mega384_crefined_{}.pth'.format(db_name))
    _, trc_trn_loss, trc_val_loss, trc_val_mmr = \
        train_contrastive(model, contrastive_ds, output_path, epochs=trn_params['epochs'], lr=trn_params['lr'],
                          loss_type=loss_type, temperature=trn_params['temperature'])

    return trc_trn_loss, trc_val_loss, trc_val_mmr



if __name__ == '__main__':

    db_subset_selection = [3]#[0, 1, 2, 3]

    for db_name in [SUBSETS[i] for i in db_subset_selection]:

        start_time = time.perf_counter()
        trc_trn_loss, trc_val_loss, trc_val_mmr = main(db_name)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        #
        fig, ax = plt.subplots(1, 1)
        ax.plot(trc_trn_loss, label='TRN {}  ({:3.1f} min.)'.format(db_name[:3], elapsed_time / 60))
        ax.plot(trc_val_loss, label='VAL ' + db_name[:3])
        if len(trc_val_mmr) > 1:
            ax.plot(2 - np.array(trc_val_mmr), label='2-MMR ' + db_name[:3])
        ax.set_ylim([0, 3])
        ax.grid(True)
        ax.legend()
        plt.show(block=False)
        plt.pause(1)
        plt.savefig(os.path.join(ROOT_MODELS, 'contrastive training convergence {}.png'.format(db_name)))
        #

    #plt.show()
