import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from wildlife_datasets.datasets import AnimalCLEF2026

from paths_and_constants import *
from image_tools import UnderwaterEnhance, TextureHarvest, AddGaussianNoise, AdaptiveCrop





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

    def attach_dataset(self, base_dataset, max_allowed_class_size=None,
                       exclude_ID_list=[], merge_IDs_not_in=[],
                       include_test=True, split_point=0.8):
        self.base_dataset = base_dataset
        self.df = base_dataset.df

        # merge IDs
        if len(merge_IDs_not_in) > 0:
            new_ID = 0#np.max(merge_IDs_not_in) + 1
            for i in range(len(self.base_dataset.labels)):
                if self.base_dataset.labels[i] in merge_IDs_not_in:
                    self.base_dataset.labels[i] = int(np.argwhere(self.base_dataset.labels[i] == merge_IDs_not_in).squeeze()) + 1
                else:
                    self.base_dataset.labels[i] = new_ID if self.base_dataset.labels[i] > -1 else self.base_dataset.labels[i]

        # 1. Build Global ID-to-Indices Map from the labels array
        self.id_to_indices = {}
        for idx, lbl in enumerate(base_dataset.labels):
            l_int = int(lbl)
            if l_int not in self.id_to_indices:
                self.id_to_indices[l_int] = []
            self.id_to_indices[l_int].append(idx)
        if not include_test:
            self.id_to_indices.pop(-1)
        for id in exclude_ID_list:
            self.id_to_indices.pop(id)

        # 2. Build Frequencies
        self.id_counts = {k: len(v) for k, v in self.id_to_indices.items()}

        # 4. Prepare the pools
        self._prepare_anchors(max_allowed_class_size=max_allowed_class_size, exclude_ID_list=exclude_ID_list, split_point=split_point)



    def _generate_all_pool_indices(self, max_id_size=None, include_test=False, exclude_ID_list=[]):
        """
        Creates a balanced or full list of indices to serve as anchors.

        Args:
            max_id_size (int, optional): The 'Chop' limit. If set, IDs with more
                                         samples will be randomly downsampled.
            include_test (bool): If False, ignores indices mapped to label -1.

        Returns:
            list: A flat list of indices for the anchor pool.
        """
        all_pool_indices = []

        # We iterate over the pre-built id_to_indices map for speed
        for lbl, indices in self.id_to_indices.items():

            # 1. Handle the test/noise split
            if lbl == -1 and not include_test:
                continue

            # 2. Handle the Exclude List
            if lbl in exclude_ID_list:
                continue

            # 3. Convert to list to allow for sampling
            #print('adding {} indices for label {}'.format(len(indices), lbl))
            current_indices = list(indices)

            # 4. Apply the 'Chop' (max_id_size)
            if max_id_size is not None and len(current_indices) > max_id_size:
                # random.sample is perfect here: it's without replacement
                # and ensures diversity within the cluster
                current_indices = random.sample(current_indices, max_id_size)

            all_pool_indices.extend(current_indices)

        # 5. Final Shuffle
        # Essential so that a single batch isn't filled with 12 images of the same ID
        random.shuffle(all_pool_indices)

        return all_pool_indices



    def _prepare_anchors(self, max_allowed_class_size=None, exclude_ID_list=[], split_point=0.8):

        labels_array = self.base_dataset.labels

        # 1. Define valid IDs for Train
        if self.trn_singletons:
            trn_valid_ids = set(self.id_to_indices.keys())
        else:
            trn_valid_ids = {id_ for id_, count in self.id_counts.items() if count > 1}

        # 2. Define valid IDs for Val
        if self.val_singletons:
            val_valid_ids = set(self.id_to_indices.keys())
        else:
            val_valid_ids = {id_ for id_, count in self.id_counts.items() if count > 1}

        # 3. Separate the pools using their specific valid_id sets
        # Note: We still use 'target_split' to find the initial training pool,
        # but for re-id we often split that pool further into trn/val anchors.
        # all_pool_indices = [
        #     i for i, (split_val, lbl) in enumerate(zip(self.df['split'], labels_array))
        #     if split_val == target_split
        # ]
        all_pool_indices = self._generate_all_pool_indices(max_id_size=max_allowed_class_size,
                                                           include_test=False, exclude_ID_list=exclude_ID_list)

        # Log the result so you can see the 'Chop' worked
        print(f"--- Anchor Pool Prepared ---")
        print(f"Total Anchors: {len(all_pool_indices)}")
        print(f"Unique IDs: {len(set(self.base_dataset.labels[all_pool_indices]))}")

        random.seed(42)
        random.shuffle(all_pool_indices)
        split_idx = int(len(all_pool_indices) * split_point)

        # Filter the split indices by their respective singleton rules
        self.trn_anchors = [i for i in all_pool_indices[:split_idx]
                            if int(labels_array[i]) in trn_valid_ids]

        self.val_anchors = [i for i in all_pool_indices[split_idx:]
                            if int(labels_array[i]) in val_valid_ids]

        # re-map IDs
        all_ancs = np.concatenate((self.trn_anchors, self.val_anchors)).astype(int)
        used_IDs = np.unique(self.base_dataset.labels[all_ancs])
        for idx, lbl in enumerate(self.base_dataset.labels):
            if lbl in used_IDs:
                self.base_dataset.labels[idx] = np.argwhere(lbl == used_IDs).squeeze()
            else:
                self.base_dataset.labels[idx] = -1
        #
        self.num_IDs = len(used_IDs)
        #
        new_id_to_indices = {}
        for new_id, old_id in enumerate(used_IDs):
            new_id_to_indices[new_id] = self.id_to_indices[old_id]
        self.id_to_indices = new_id_to_indices


        print(f"Prepared {len(self.trn_anchors)} trn and {len(self.val_anchors)} val anchors.")



    def get_num_IDs(self):

        return self.num_IDs



    def enable_singletons(self, trn_enabled: bool, val_enabled: bool):
        """
        Independent control for singletons.
        Commonly: trn_enabled=True (for more data) and val_enabled=False (for cleaner metrics).
        """
        self.trn_singletons = trn_enabled
        self.val_singletons = val_enabled

        if self.base_dataset is not None:
            self._prepare_anchors()
            # Refresh the active anchor pointer
            self.use_split(self.split)


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


    ###############################################################################
    #
    # THE FOLLOWING METHOD INTENDED TO SUPPORT NON-TRIVIAL SAMPLING OF THE DATA
    #
    ###############################################################################

    def __get_id_frequencies(self):
        """
        Returns the unique IDs currently active in the current split
        and the frequency (count) of anchors associated with each ID.
        """
        if self.anchors is None or len(self.anchors) == 0:
            return [], []

        # Get the labels for only the currently active anchors
        # base_dataset.labels is the ndarray of ints
        active_labels = self.base_dataset.labels[self.anchors]

        # Use numpy to get unique IDs and their counts in the active set
        ids, freq = np.unique(active_labels, return_counts=True)

        return ids.tolist(), freq.tolist()

    def get_sample_frequency(self):

        id_list, id_freq = self.__get_id_frequencies()
        anc_labels = [int(self.base_dataset.labels[i]) for i in self.anchors]
        id_to_freq = dict(zip(id_list, id_freq))
        anc_freqs = [id_to_freq[lbl] for lbl in anc_labels]

        return anc_freqs


    def get_anchor_idxs_and_ids(self):
        """Returns (indices, labels) for the active split."""
        if self.anchors is None:
            return [], []

        # Get the labels for these specific indices
        active_labels = [int(self.base_dataset.labels[i]) for i in self.anchors]

        # Both self.anchors and active_labels are now the same length
        return self.anchors, active_labels






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
        T.ToTensor(),
        T.Resize((640, 640)),
        AdaptiveCrop(),
        UnderwaterEnhance(),  # First, fix the visibility
        T.Resize((384, 384)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.12, contrast=0.12),
        #T.ToTensor(),
        # TextureHarvest(),
        # AddGaussianNoise(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = T.Compose([
        T.ToTensor(),
        T.Resize((640, 640)),
        AdaptiveCrop(),
        UnderwaterEnhance(),  # First, fix the visibility
        T.Resize((384, 384)),
        #T.ToTensor(),
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
        triplet_ds.enable_singletons(trn_enabled=True, val_enabled=True)
        triplet_ds.attach_dataset(base_dataset)#, merge_IDs_not_in=[1, 26, 5, 8, 10, 23, 21, 14, 27])  # 'base_dataset' is your original class
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