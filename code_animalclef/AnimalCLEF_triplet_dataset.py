import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from wildlife_datasets.datasets import AnimalCLEF2026

from paths_and_constants import *
from image_tools import UnderwaterEnhance


# class AnimalCLEFTripletDataset(Dataset):
#     def __init__(self):
#         self.df = None
#         self.base_dataset = None
#         self.transform = None
#         self.trn_transform = None
#         self.val_transform = None
#         self.split = 'trn'
#         self.anchors = []
#         self.id_to_indices = {}
#
#     def attach_dataset(self, base_dataset):
#         """
#         Wraps an existing AnimalCLEF dataset.
#         Expects base_dataset.df to have ['individual_id', 'is_train']
#         """
#         self.base_dataset = base_dataset
#         self.df = base_dataset.df
#
#         # 1. Separate TRN and VAL (assuming is_train is a boolean or 0/1)
#         # You can adjust this logic if your split is handled differently
#         self.full_train_df = self.df[self.df['split'] == 'train'].copy()
#
#         # 2. Map every ID to its list of row indices for fast lookup
#         self.id_to_indices = self.full_train_df.groupby('identity').indices
#
#         # 3. Filter Anchors: Only IDs with > 1 example are valid
#         self.valid_ids = [id_ for id_, idxs in self.id_to_indices.items() if len(idxs) > 1]
#
#         # 4. Create a consistent list of all possible anchor images
#         # An anchor is any image belonging to a valid_id
#         self.all_anchor_indices = self.full_train_df[
#             self.full_train_df['identity'].isin(self.valid_ids)
#         ].index.tolist()
#
#         # 5. Simple 80/20 split for TRN/VAL
#         random.seed(42)
#         random.shuffle(self.all_anchor_indices)
#         split_idx = int(len(self.all_anchor_indices) * 0.8)
#
#         self.trn_anchors = self.all_anchor_indices[:split_idx]
#         self.val_anchors = self.all_anchor_indices[split_idx:]
#
#         print(f"Dataset Attached: {len(self.trn_anchors)} TRN anchors, {len(self.val_anchors)} VAL anchors.")
#
#     def config_trn_transforms(self, transforms):
#         self.trn_transform = transforms
#
#     def config_val_transforms(self, transforms):
#         self.val_transform = transforms
#
#     def use_split(self, split):
#         self.split = split
#         self.anchors = self.trn_anchors if split == 'trn' else self.val_anchors
#         self.transform = self.trn_transform if split == 'trn' else self.val_transform
#
#     def __len__(self):
#         return len(self.anchors)
#
#     def __getitem__(self, idx):
#         # 1. Get Anchor
#         anchor_idx = self.anchors[idx]
#         anchor_row = self.df.loc[anchor_idx]
#         anchor_id = anchor_row['identity']
#
#         # 2. Pick Positive (Same ID, different image)
#         pos_options = self.id_to_indices[anchor_id]
#         # Filter out the anchor itself from options
#         pos_options = [i for i in pos_options if i != anchor_idx]
#         pos_idx = random.choice(pos_options)
#
#         # 3. Pick Negative (Different ID)
#         # We pick a random ID that isn't the anchor_id
#         neg_id = random.choice(list(self.id_to_indices.keys()))
#         while neg_id == anchor_id:
#             neg_id = random.choice(list(self.id_to_indices.keys()))
#
#         neg_idx = random.choice(self.id_to_indices[neg_id])
#
#         # 4. Load and Transform images
#         # Assuming your base_dataset has a method to load image by index
#         anchor_img = self.base_dataset.get_image(anchor_idx)
#         pos_img = self.base_dataset.get_image(pos_idx)
#         neg_img = self.base_dataset.get_image(neg_idx)
#
#         if self.transform:
#             anchor_img = self.transform(anchor_img)
#             pos_img = self.transform(pos_img)
#             neg_img = self.transform(neg_img)
#
#         return {
#             'anchor': anchor_img,
#             'positive': pos_img,
#             'negative': neg_img,
#             'anchor_id': anchor_id,
#             'neg_id': neg_id
#         }






class AnimalCLEFTripletDataset(Dataset):

    def __init__(self):
        self.base_dataset = None
        self.df = None
        self.include_singletons = False

        # Identity is already factorized into integers
        self.id_to_indices = None

        # Storage for the two transform pipelines
        self.trn_transform = None
        self.val_transform = None
        self.split = 'trn'

    def attach_dataset(self, base_dataset):
        self.base_dataset = base_dataset
        self.df = base_dataset.df

        # 1. Build Global ID-to-Indices Map from the labels array
        self.id_to_indices = {}
        for idx, lbl in enumerate(base_dataset.labels):
            l_int = int(lbl)
            if l_int not in self.id_to_indices:
                self.id_to_indices[l_int] = []
            self.id_to_indices[l_int].append(idx)

        # 2. Build Frequencies
        self.id_counts = {k: len(v) for k, v in self.id_to_indices.items()}

        # 3. Prepare the pools
        self._prepare_anchors()

    def _prepare_anchors(self):
        # Use the clean integers from the base_dataset.labels ndarray
        labels_array = self.base_dataset.labels

        # Identify which split name to use (e.g., 'train')
        actual_splits = self.df['split'].unique()
        target_split = 'train' if 'train' in actual_splits else actual_splits[0]

        # Filter IDs based on singleton policy (using the global map we built)
        if self.include_singletons:
            valid_ids = set(self.id_to_indices.keys())
        else:
            valid_ids = {id_ for id_, count in self.id_counts.items() if count > 1}

        # Create the anchor pool by checking both the split and the ID
        # We iterate indices to keep it aligned with base_dataset.labels
        self.all_anchor_indices = [
            i for i, (split_val, lbl) in enumerate(zip(self.df['split'], labels_array))
            if split_val == target_split and int(lbl) in valid_ids
        ]

        # Perform the 80/20 split on the pool
        random.seed(42)
        random.shuffle(self.all_anchor_indices)
        split_idx = int(len(self.all_anchor_indices) * 0.8)

        self.trn_anchors = self.all_anchor_indices[:split_idx]
        self.val_anchors = self.all_anchor_indices[split_idx:]

        print(f"Prepared {len(self.trn_anchors)} train and {len(self.val_anchors)} val anchors.")



    def enable_singletons(self, enabled: bool):
        """
        Toggle whether to include singleton IDs (IDs with only 1 image).
        If enabled, the positive image for a singleton anchor will be an
        augmented version of the anchor itself.
        """
        self.include_singletons = enabled
        # If the dataset is already attached, we refresh the anchor lists to reflect the change
        if self.base_dataset is not None:
            self.attach_dataset(self.base_dataset)



    def config_trn_transforms(self, transforms):
        self.trn_transform = transforms

    def config_val_transforms(self, transforms):
        self.val_transform = transforms



    def use_split(self, split):
        """The 'Switch': Updates the base dataset's transform dynamically."""
        self.split = split
        self.anchors = self.trn_anchors if split == 'trn' else self.val_anchors

        # Dynamically update the transform on the base object
        new_tf = self.trn_transform if split == 'trn' else self.val_transform
        if hasattr(self.base_dataset, 'set_transform'):
            self.base_dataset.set_transform(new_tf)
        else:
            # Fallback if the specific version uses a direct attribute
            self.base_dataset.transform = new_tf

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        anchor_idx = self.anchors[idx]

        # Delegation: Base dataset handles the current transform and int labels
        anchor_img, anchor_label = self.base_dataset[anchor_idx]
        anchor_label = int(anchor_label)

        # Positive selection logic
        all_pos_indices = self.id_to_indices[anchor_label]
        other_pos_indices = [i for i in all_pos_indices if i != anchor_idx]
        pos_idx = random.choice(other_pos_indices) if other_pos_indices else anchor_idx
        pos_img, assert_pos_label = self.base_dataset[pos_idx]

        # Negative selection logic
        neg_label = random.choice(list(self.id_to_indices.keys()))
        while neg_label == anchor_label:
            neg_label = random.choice(list(self.id_to_indices.keys()))

        neg_idx = random.choice(self.id_to_indices[neg_label])
        neg_img, assert_neg_label = self.base_dataset[neg_idx]

        assert anchor_label == assert_pos_label
        assert anchor_label != assert_neg_label

        return {
            'anchor': anchor_img,
            'positive': pos_img,
            'negative': neg_img,
            'anchor_id': torch.tensor(anchor_label, dtype=torch.long),
            'neg_id': torch.tensor(neg_label, dtype=torch.long)
        }



if __name__ == '__main__':

    import torch
    import torchvision.transforms as T
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # --- 1. Define typical transformations for Re-ID ---
    # Since you're working with 384x384 (Mega-384), we use that size.
    train_transforms = T.Compose([
        UnderwaterEnhance,  # First, fix the visibility
        T.Resize((384, 384)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.12, contrast=0.12),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = T.Compose([
        UnderwaterEnhance,  # First, fix the visibility
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    def main():
        # --- 2. Initialize the Base Dataset ---
        # Assuming your local AnimalCLEF2026 class exists:
        dataset_full = AnimalCLEF2026(
            ROOT_DATA,
            transform=None,
            load_label=True,  # return label as 2nd parameter
            factorize_label=True,  # replace string for unique integer
            check_files=False
        )
        base_dataset = dataset_full.get_subset(dataset_full.df['dataset'] == SUBSETS[2])

        print("Initializing Triplet Dataset...")
        triplet_ds = AnimalCLEFTripletDataset()
        triplet_ds.enable_singletons(True)
        triplet_ds.attach_dataset(base_dataset)  # 'base_dataset' is your original class
        #triplet_ds.enable_singletons(True)

        # --- 3. Configure Splits and Transforms ---
        triplet_ds.config_trn_transforms(train_transforms)
        triplet_ds.config_val_transforms(val_transforms)


        # Switch to TRN mode
        triplet_ds.use_split('trn')

        # --- 4. Create DataLoader ---
        # Since you're on an external drive, keep num_workers low initially to avoid lag
        loader = DataLoader(triplet_ds, batch_size=4, shuffle=True, num_workers=2)

        # --- 5. Fetch and Visualize a Batch ---
        batch = next(iter(loader))

        anchors = batch['anchor']  # [B, 3, 384, 384]
        positives = batch['positive']
        negatives = batch['negative']

        print(f"Batch loaded. Shape: {anchors.shape}")
        print(f"IDs in this batch: {batch['anchor_id']}")

        # --- 6. Visualization Plot ---
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        plt.suptitle("Sea Turtle Triplets (Anchor | Positive | Negative)", fontsize=16)

        for i in range(4):
            # Helper to un-normalize for viewing
            def denorm(img):
                img = img.permute(1, 2, 0).cpu().numpy()
                img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                return img.clip(0, 1)

            axes[i, 0].imshow(denorm(anchors[i]))
            axes[i, 0].set_title(f"A: {batch['anchor_id'][i]}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(denorm(positives[i]))
            axes[i, 1].set_title(f"P: {batch['anchor_id'][i]}")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(denorm(negatives[i]))
            axes[i, 2].set_title(f"N: {batch['neg_id'][i]}")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.show()



    main()