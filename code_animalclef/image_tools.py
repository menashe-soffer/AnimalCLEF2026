import cv2
import numpy as np
from PIL import Image
import torch
import random



def apply_clahe(img):
    # Convert PIL Image to OpenCV (BGR) format
    img_np = np.array(img)

    # Convert to LAB color space. We only want to touch the L (Lightness) channel.
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Define CLAHE parameters.
    # ClipLimit=2.0 (Lower means less aggressive contrast, better for noise)
    # TileGridSize=(8,8) (How many local regions to process independently)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE only to the Lightness channel
    cl = clahe.apply(l)

    # Merge channels back together and convert back to RGB
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Convert back to PIL for the transform pipeline
    return Image.fromarray(enhanced_img)



def UnderwaterEnhance(img, clip_limit=2.0-0.5, tile_size=(8, 8)):
    # 1. Convert PIL to OpenCV (NumPy)
    img_np = np.array(img)

    # 2. Move to LAB color space
    # L = Lightness, A/B = Color dimensions
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 3. Create and apply CLAHE to the 'L' channel only
    # This enhances contrast without shifting the colors
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    cl = clahe.apply(l)

    # 4. Merge back and convert to RGB
    limg = cv2.merge((cl, a, b))
    enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # 5. Return as PIL Image so the next transform can process it
    return Image.fromarray(enhanced_np)






###########################################
#
#  the following are designed to combat Lynx dataset flaws
#
###########################################

class RandomLynxJigsaw(torch.nn.Module):
    """
    Splits image into 64x64 tiles, picks the 18 most 'filled' tiles,
    and scatters them (doubled) into a 384x384 canvas.
    """

    def __init__(self, p=0.3, tile_size=64, num_best=18):
        super().__init__()
        self.p = p
        self.tile_size = tile_size
        self.num_best = num_best

    def forward(self, img):
        if random.random() > self.p:
            return img

        C, H, W = img.shape
        # 1. Slice into tiles [C, 6, 6, 64, 64]
        tiles = img.unfold(1, self.tile_size, self.tile_size).unfold(2, self.tile_size, self.tile_size)

        # 2. Score tiles by content (sum of absolute values to handle normalized mean)
        # We want tiles that aren't just 'background'
        scores = tiles.abs().sum(dim=(0, 3, 4))  # [6, 6]
        flat_scores = scores.view(-1)

        # 3. Get the top N indices
        _, top_idx = torch.topk(flat_scores, self.num_best)

        # 4. Extract those tiles [18, C, 64, 64]
        flat_tiles = tiles.permute(1, 2, 0, 3, 4).reshape(-1, C, self.tile_size, self.tile_size)
        best_tiles = flat_tiles[top_idx]

        # 5. Double them and shuffle for the 36 slots
        jigsaw_pieces = torch.cat([best_tiles, best_tiles], dim=0)
        jigsaw_pieces = jigsaw_pieces[torch.randperm(36)]

        # 6. Reassemble into a new canvas
        new_img = torch.zeros_like(img)
        count = 0
        for i in range(0, H, self.tile_size):
            for j in range(0, W, self.tile_size):
                new_img[:, i:i + self.tile_size, j:j + self.tile_size] = jigsaw_pieces[count]
                count += 1
        return new_img


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.04, p=0.5):
        super().__init__()
        self.std = std
        self.mean = mean
        self.p = p

    def forward(self, tensor):
        if random.random() > self.p:
            return tensor
        # Adds noise scaled by the standard deviation
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean


class TextureHarvest(torch.nn.Module):
    def __init__(self, target_size=320, tile_size=64):
        super().__init__()
        self.target_size = target_size
        self.tile_size = tile_size

    def forward(self, img):
        # img is now a Tensor [C, 512, 512]
        C, H, W = img.shape
        num_slots_axis = self.target_size // self.tile_size
        total_slots = num_slots_axis ** 2

        # 1. Slice into 64x64 tiles
        tiles = img.unfold(1, self.tile_size, self.tile_size).unfold(2, self.tile_size, self.tile_size)

        # 2. Score by intensity (finding the Lynx fur)
        # Note: At this stage, img is [0, 1]. Summing works well.
        scores = tiles.sum(dim=(0, 3, 4))
        flat_scores = scores.view(-1)

        # 3. Pick the best N tiles
        _, top_idx = torch.topk(flat_scores, total_slots)
        flat_tiles = tiles.permute(1, 2, 0, 3, 4).reshape(-1, C, self.tile_size, self.tile_size)
        best_tiles = flat_tiles[top_idx]

        # 4. Shuffle them so the model doesn't learn a "fixed" jigsaw
        best_tiles = best_tiles[torch.randperm(total_slots)]

        # 5. Build the smaller canvas (320x320)
        new_img = torch.zeros((C, self.target_size, self.target_size), device=img.device)
        k = 0
        for i in range(num_slots_axis):
            for j in range(num_slots_axis):
                r_start, c_start = i * self.tile_size, j * self.tile_size
                new_img[:, r_start:r_start + self.tile_size, c_start:c_start + self.tile_size] = best_tiles[k]
                k += 1
        return new_img
