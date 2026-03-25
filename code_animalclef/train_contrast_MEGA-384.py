import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import os
import gc
import time
from wildlife_datasets.datasets import AnimalCLEF2026

from paths_and_constants import *
from AnimalCLEF_contrastive_dataset import AnimalCLEFContrastiveDataset  # Your new class
from image_tools import UnderwaterEnhance
from monitoring import print_vram_stats


class SupConLoss(nn.Module):
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


import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
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



class AnimalReIDRefiner(nn.Module):
    def __init__(self, model_name="hf-hub:BVRA/MegaDescriptor-L-384", use_projector=True, projection_dim=256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone.reset_classifier(0)

        # 4. Add the SimCLR/SupCon standard projection head
        feature_dim = self.backbone.num_features
        if use_projector:
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, projection_dim)  # Projects to a 128-D unit sphere
            )
        else:
            self.projector = nn.Identity()


    def freeze_for_training(self, active_stages=[3]):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for stage_idx in active_stages:
            for param in self.backbone.layers[stage_idx].parameters():
                param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
        for param in self.backbone.head.parameters():
            param.requires_grad = True

    def forward(self, x):

        features = self.backbone(x)
        return self.projector(features)


# option: add projection head
#
# class AnimalReIDRefiner(nn.Module):
#     def __init__(self, model_name="hf-hub:BVRA/MegaDescriptor-L-384", use_projector=True, projection_dim=128):
#         super().__init__()
#         self.model = timm.create_model(model_name, pretrained=True)
#         self.use_projector = use_projector
#
#         # Swin-L (MegaDescriptor) output dim is 1536
#         self.in_features = self.model.num_features
#
#         # Replace the original classification head with Identity to get the raw 1536 vector
#         self.model.reset_classifier(0)
#
#         if self.use_projector:
#             # The Non-Linear Projector (MLP)
#             self.projector = nn.Sequential(
#                 nn.Linear(self.in_features, self.in_features),
#                 nn.ReLU(),
#                 nn.Linear(self.in_features, projection_dim)
#             )
#         else:
#             self.projector = nn.Identity()
#
#     def freeze_for_training(self, active_stages=[3]):
#         # 1. Freeze everything
#         for param in self.parameters():
#             param.requires_grad = False
#
#         # 2. Unfreeze specific Swin stages (blocks)
#         for stage_idx in active_stages:
#             for param in self.model.layers[stage_idx].parameters():
#                 param.requires_grad = True
#
#         # 3. Unfreeze final norm
#         for param in self.model.norm.parameters():
#             param.requires_grad = True
#
#         # 4. ALWAYS unfreeze the projector (or Identity head)
#         if self.use_projector:
#             for param in self.projector.parameters():
#                 param.requires_grad = True
#
#     def forward(self, x):
#         features = self.model(x)  # 1536-dim embedding
#         return self.projector(features)  # 128-dim projection




def train_step(model, loader, device, optimizer, criterion, pbar_str):

    running_loss = 0.0
    # print_vram_stats()

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
        features = torch.nn.functional.normalize(features, dim=1)
        multi_labels = torch.cat([labels, labels], dim=0)
        if optimizer is not None:
            optimizer.zero_grad()

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


    return  avg_loss


def train_contrastive(model, dataset, output_fname, epochs=5, lr=1e-4, batch_size=8, loss_type='SupCon', device='cuda'):

    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    if loss_type == 'SupCon':
        criterion = SupConLoss(temperature=0.07)
    if loss_type == 'NTXent':
        criterion = NTXentLoss(temperature=0.07)

    best_loss = 999
    trc_trn_loss, trc_val_loss = [], []

    for epoch in range(epochs):

        # train step
        model.train()
        dataset.set_trn()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        avg_trn_loss = train_step(model, loader, device, optimizer, criterion,
                              pbar_str=f"Epoch {epoch + 1}/{epochs} [{loss_type}]")

        # val step
        model.eval()
        dataset.set_val()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        avg_val_loss = train_step(model, loader, device, None, criterion,
                              pbar_str=f"Epoch {epoch + 1}/{epochs} [{loss_type}] (VAL)")
        print(f"Epoch {epoch + 1} Complete | Avg Loss: TRN: {avg_trn_loss:.4f}  VAL: {avg_val_loss:.4f}")
        trc_trn_loss.append(avg_trn_loss)
        trc_val_loss.append(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save state_dict (we'll remember to strip 'model.' later if needed)
            torch.save(model.state_dict(), output_fname)
            print(f"saving weights to {output_fname}")

    return model, trc_trn_loss, trc_val_loss


# for Labeled SupCon
weak_train_transforms = T.Compose([
    UnderwaterEnhance,
    T.RandomResizedCrop(384, scale=(0.75, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


TRN_PARAMS = dict()
TRN_PARAMS['SalamanderID2025'] = dict({'epochs': 7, 'transform': None, 'loss_type': 'NTXent', 'temperature': 0.07, 'val_ratio': 0.2})
TRN_PARAMS['SeaTurtleID2022'] = dict({'epochs': 5, 'transform': weak_train_transforms, 'loss_type': 'SupCon', 'temperature': 0.07, 'val_ratio': 0.2})
TRN_PARAMS['LynxID2025'] = dict({'epochs': 5, 'transform': weak_train_transforms, 'loss_type': 'SupCon', 'temperature': 0.07, 'val_ratio': 0.2})
TRN_PARAMS['TexasHornedLizards'] = dict({'epochs': 6, 'transform': None, 'loss_type': 'NTXent', 'temperature': 0.15, 'val_ratio': 0.2})



def main(db_name):

    trn_params = TRN_PARAMS[db_name]
    print('\n\ntraining with dataset', db_name)

    loss_type = trn_params['loss_type']   # 'SupCon', 'NTXent'

    # 1. Setup Base Data
    dataset_full = AnimalCLEF2026(root=ROOT_DATA)
    # This subset can now include your 'test' split for lizards!
    base_subset = dataset_full.get_subset(dataset_full.df['dataset'] == db_name)

    # 2. Setup Contrastive Dataset
    contrastive_ds = AnimalCLEFContrastiveDataset()
    contrastive_ds.attach_dataset(base_subset)
    contrastive_ds.make_split(val_ratio=trn_params['val_ratio'])

    contrastive_ds.config_transforms(transforms_trn=trn_params['transform']) # set default for now

    # 4. Model and Training
    model = AnimalReIDRefiner()
    model.freeze_for_training(active_stages=[3])

    output_path = os.path.join(ROOT_MODELS,'mega384_refined_{}_{}.pth'.format(loss_type, db_name))
    _, trc_trn_loss, trc_val_loss = \
        train_contrastive(model, contrastive_ds, output_path, epochs=trn_params['epochs'], loss_type=loss_type)#, batch_size=16)

    return trc_trn_loss, trc_val_loss



if __name__ == '__main__':

    db_subset_selection = [0, 1, 2, 3]

    for db_name in [SUBSETS[i] for i in db_subset_selection]:

        start_time = time.perf_counter()
        trc_trn_loss, trc_val_loss = main(db_name)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        #
        plt.plot(trc_trn_loss, label='TRN {}  ({:3.1f} min.)'.format(db_name[:3], elapsed_time / 60))
        plt.plot(trc_val_loss, label='VAL ' + db_name[:3])
        plt.ylim([0, 5])
        plt.grid(True)
        plt.legend()
        plt.show(block=False)
        plt.pause(1)
        plt.savefig(os.path.join(ROOT_MODELS, 'contrastive training convergence.png'))
        #

    plt.show()
