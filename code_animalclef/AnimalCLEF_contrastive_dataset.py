import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import random
from wildlife_datasets.datasets import AnimalCLEF2026

from paths_and_constants import *
from image_tools import UnderwaterEnhance


class AnimalCLEFContrastiveDataset(Dataset):
    def __init__(self):
        self.df = None
        self.base_dataset = None
        self.transform = None
        self.id_to_indices = {}

    def attach_dataset(self, base_dataset):
        """
        Wraps an existing AnimalCLEF dataset.
        Handles both labeled (train) and unlabeled (test) splits.
        """
        self.base_dataset = base_dataset
        self.df = base_dataset.df.copy()

        # 1. Map existing labels and create unique IDs for 'test' split
        # We assume 'identity' contains the labels and 'split' identifies test data

        self.id_to_indices = self.df.groupby('identity').indices
        labels = - np.ones_like(self.df['identity'].values)
        for id, key in enumerate(self.id_to_indices.keys()):
            for idx in self.id_to_indices[key]:
                labels[idx] = id
        self.last_genuine_id = labels.max()
        for idx in np.argwhere(labels == -1):
            labels[idx] = labels.max() + 1

        self.df['assigned_label'] = labels
        self.id_to_indices = self.df.groupby('assigned_label').indices

        print(f"Dataset Attached: {len(self.df)} total images.")
        #print(f"Labeled IDs: {max_labeled_id + 1}, Unlabeled (Unique) IDs: {num_test}")



    def is_genuine(self, label):

        return label <= self.last_genuine_id



    def config_transforms(self, transforms_trn=None, transforms_val=None, enhance=True):
        """
        In SimCLR, we usually use the same stochastic transform pipeline
        for both 'q' and 'k' versions.
        """

        if transforms_trn is None:
            # set default transform
            self.transform_trn = T.Compose([
                # UnderwaterEnhance,
                T.Resize((384, 384)),
                T.RandomResizedCrop(size=384, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            if enhance:
                self.transform_trn = T.Compose([UnderwaterEnhance, self.transform_trn])
        else:
            self.transform_trn = transforms_trn

        if transforms_val is None:
            # set default transform
            self.transform_val = T.Compose([
                # UnderwaterEnhance,  # First, fix the visibility
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            if enhance:
                self.transform_val = T.Compose([UnderwaterEnhance, self.transform_val])
        else:
            self.transform_val = transforms_val

    def make_split(self, val_ratio=0.2, seed=42):
        """
        Splits by Identity using the labels generated in attach_dataset.
        """
        random.seed(seed)

        # 1. Get unique assigned labels from your existing logic
        unique_ids = self.df['assigned_label'].unique()
        random.shuffle(unique_ids)

        # 2. Determine which IDs go to Val
        split_idx = int(len(unique_ids) * (1 - val_ratio))
        train_ids = set(unique_ids[:split_idx])
        val_ids = set(unique_ids[split_idx:])

        # 3. Create the two dataframes
        # We keep the original df as 'df_total' to allow re-splitting if needed
        if not hasattr(self, 'df_total'):
            self.df_total = self.df.copy()

        self.trn_df = self.df_total[self.df_total['assigned_label'].isin(train_ids)].copy()
        self.val_df = self.df_total[self.df_total['assigned_label'].isin(val_ids)].copy()

        print(f"Split completed: {len(train_ids)} IDs in TRN, {len(val_ids)} IDs in VAL.")

    def set_trn(self):
        """Points the dataset to the training split."""
        self.df = self.trn_df
        # Re-run your existing grouping logic so __getitem__ finds the right indices
        self.id_to_indices = self.df.groupby('assigned_label').indices
        print(f"Dataset set to TRAIN: {len(self.df)} images.")
        self.transform = self.transform_trn

    def set_val(self):
        """Points the dataset to the validation split."""
        self.df = self.val_df
        # Re-run your existing grouping logic
        self.id_to_indices = self.df.groupby('assigned_label').indices
        print(f"Dataset set to VAL: {len(self.df)} images.")
        self.transform = self.transform_val



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Anchor
        anchor_row = self.df.iloc[idx]
        anchor_id = anchor_row['assigned_label']
        anchor_img = self.base_dataset.get_image(idx)

        # 2. Find a "Partner" (Same ID, different image)
        all_indices_with_id = np.argwhere(anchor_id == self.df['assigned_label']).flatten()#self.id_to_indices[anchor_id]
        partner_options = [i for i in all_indices_with_id if i != idx]

        if len(partner_options) > 0:
            # CASE A: LABELED - Pick a REAL different image of the same individual
            partner_idx = random.choice(partner_options)
            partner_img = self.base_dataset.get_image(partner_idx)

            # Apply random transforms to both distinct images
            im_q = self.transform(anchor_img)
            im_k = self.transform(partner_img)
        else:
            # CASE B: UNLABELED/UNIQUE - Two different augmentations of the SAME image
            im_q = self.transform(anchor_img)
            im_k = self.transform(anchor_img)

        return {
            'im_q': im_q,
            'im_k': im_k,
            'label': anchor_id
        }


if __name__ == '__main__':

    import torchvision.transforms as T
    from torch.utils.data import DataLoader


    def main():

        # --- 2. Initialize and Attach ---
        # (Assuming your base_dataset setup from before)
        dataset_full = AnimalCLEF2026(root=ROOT_DATA)
        base_dataset = dataset_full.get_subset(dataset_full.df['dataset'] == SUBSETS[1])

        contrastive_ds = AnimalCLEFContrastiveDataset()
        contrastive_ds.attach_dataset(base_dataset)
        contrastive_ds.config_transforms()
        contrastive_ds.make_split()
        contrastive_ds.set_trn()
        print(contrastive_ds.__len__(), 'items in trn set')
        contrastive_ds.set_val()
        print(contrastive_ds.__len__(), 'items in val set')

        # --- 3. Create DataLoader ---
        # Note: Batch size N results in 2N images in the training loop
        loader = DataLoader(contrastive_ds, batch_size=8, shuffle=True, num_workers=4)

        fig, ax = plt.subplots(4, 4)
        ax[0, 0].set_title('TRN (P1)')
        ax[0, 1].set_title('TRN (P2)')
        ax[0, 2].set_title('VAL (P1)')
        ax[0, 3].set_title('VAL (P2)')
        [ax_.axis('off') for ax_ in ax.flatten()]

        def denorm(img):
            img = img.permute(1, 2, 0).cpu().numpy()
            img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            return img.clip(0, 1)

        contrastive_ds.set_trn()

        batch = next(iter(loader))
        im_q = batch['im_q']  # [8, 3, 384, 384]
        im_k = batch['im_k']  # [8, 3, 384, 384]
        labels = batch['label']

        for i in range(min(4, im_q.shape[0])):
            ax[i, 0].imshow(denorm(im_q[i]))
            ax[i, 1].imshow(denorm(im_k[i]))

        contrastive_ds.set_val()

        batch = next(iter(loader))
        im_q = batch['im_q']  # [8, 3, 384, 384]
        im_k = batch['im_k']  # [8, 3, 384, 384]
        labels = batch['label']

        for i in range(min(4, im_q.shape[0])):
            ax[i, 2].imshow(denorm(im_q[i]))
            ax[i, 3].imshow(denorm(im_k[i]))

        plt.show()


    main()