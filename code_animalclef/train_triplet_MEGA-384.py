import torch
import torch.nn as nn
import timm
import torchvision.transforms as T

from wildlife_datasets.datasets import AnimalCLEF2026

from paths_and_constants import *
from AnimalCLEF_triplet_dataset import AnimalCLEFTripletDataset
from image_tools import UnderwaterEnhance


class AnimalReIDRefiner(nn.Module):

    def __init__(self, model_name="hf-hub:BVRA/MegaDescriptor-L-384"):
        super().__init__()
        # Load the model with its original pretrained head
        self.model = timm.create_model(model_name, pretrained=True)

    def freeze_for_training(self, active_stages=[3]):
        """
        Freezes the backbone except for chosen stages.
        Swin-L has 4 stages (index 0 to 3).
        Example: [2, 3] unfreezes the last two hierarchical stages.
        """
        # 1. Freeze everything initially
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Unfreeze specific stages in the Swin backbone
        # In timm, these are stored in self.model.layers
        for stage_idx in active_stages:
            print(f"Unfreezing Stage {stage_idx + 1}...")
            for param in self.model.layers[stage_idx].parameters():
                param.requires_grad = True

        # 3. Always unfreeze the final norm and the head
        for param in self.model.norm.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        # returns the embedding (original head output)
        return self.model(x)


import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


def train_model(model, dataset, output_fname, epochs=10, lr=1e-4, batch_size=8, device='cuda'):

    model.to(device)

    # Only optimize parameters that are NOT frozen (Stages 3/4 + Head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Standard Triplet Margin Loss
    criterion = torch.nn.TripletMarginLoss(margin=0.3*1.5, p=2)

    best_avg_val = 999
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        dataset.use_split('trn')
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        model.train()
        running_trn_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} [TRN]")
        for batch in pbar:
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)

            optimizer.zero_grad()

            # Forward passes
            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)

            loss = criterion(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()

            running_trn_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- VALIDATION PHASE ---
        dataset.use_split('val')
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                a_emb = model(batch['anchor'].to(device))
                p_emb = model(batch['positive'].to(device))
                n_emb = model(batch['negative'].to(device))

                loss = criterion(a_emb, p_emb, n_emb)
                running_val_loss += loss.item()

        avg_trn = running_trn_loss / len(loader)
        avg_val = running_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1} Complete | TRN Loss: {avg_trn:.4f} | VAL Loss: {avg_val:.4f}")

        if avg_val < best_avg_val:

            # Save the specific weights
            best_avg_val = avg_val
            torch.save(model.state_dict(), output_fname)

    return model




def main(subset_id, num_epochs=5):


    # (1) Generate and configure models
    dataset_full = AnimalCLEF2026(root=ROOT_DATA)
    base_dataset = dataset_full.get_subset(dataset_full.df['dataset'] == SUBSETS[subset_id])
    triplet_ds = AnimalCLEFTripletDataset()
    triplet_ds.attach_dataset(base_dataset=base_dataset)

    # We use your generic class from before
    model = AnimalReIDRefiner()
    model.freeze_for_training(active_stages=[3])

    # (2) define transforms
    # Since you're working with 384x384 (Mega-384), we use that size.
    train_transforms = T.Compose([
        UnderwaterEnhance,  # First, fix the visibility
        T.Resize((384, 384)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = T.Compose([
        UnderwaterEnhance,  # First, fix the visibility
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Configure transforms (using the ones we discussed earlier)
    triplet_ds.config_trn_transforms(train_transforms)
    triplet_ds.config_val_transforms(val_transforms)

    # (4) Refine Specifically for Turtles
    # You might want to filter the base_ds to turtles ONLY before this step
    print("\n--- Starting {} Refinement ---".format(SUBSETS[subset_id]))
    output_fname = os.path.join(ROOT_MODELS,'mega384_refined_{}_trplt.pth'.format(SUBSETS[subset_id]))
    refined_model = train_model(model, triplet_ds, output_fname, epochs=num_epochs)

    # # Save the specific weights
    # torch.save(refined_model.state_dict(), os.path.join(ROOT_MODELS,'mega384_refined_{}.pth'.format(SUBSETS[subset_id])))




if __name__ == '__main__':

    main(subset_id=0, num_epochs=12)
    #main(subset_id=1, num_epochs=7)
    main(subset_id=2, num_epochs=12)

