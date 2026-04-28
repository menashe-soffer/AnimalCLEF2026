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

#
#
# def UnderwaterEnhance(img, clip_limit=2.0-0.5, tile_size=(8, 8)):
#     # 1. Convert PIL to OpenCV (NumPy)
#     img_np = np.array(img)
#
#     # 2. Move to LAB color space
#     # L = Lightness, A/B = Color dimensions
#     lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#
#     # 3. Create and apply CLAHE to the 'L' channel only
#     # This enhances contrast without shifting the colors
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
#     cl = clahe.apply(l)
#
#     # 4. Merge back and convert to RGB
#     limg = cv2.merge((cl, a, b))
#     enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
#
#     # 5. Return as PIL Image so the next transform can process it
#     return Image.fromarray(enhanced_np)
#
#
import numpy as np
import cv2
import torch
from torchvision.transforms import functional as F


class UnderwaterEnhance:
    def __init__(self, clip_limit=1.5, tile_size=(8, 8)):
        #self.clip_limit = clip_limit
        try:
            self.clip_limit = float(clip_limit)
        except (TypeError, ValueError):
            self.clip_limit = 1.5  # Fallback to default
        self.tile_size = tile_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_size)

    def __call__(self, img_tensor):
        # 1. Tensor (C, H, W) -> NumPy (H, W, C)
        # We multiply by 255 because OpenCV expects uint8 for LAB conversion
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # 2. LAB Conversion & CLAHE (Standard OpenCV logic)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)

        # 3. Merge and back to RGB
        limg = cv2.merge((cl, a, b))
        enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # 4. NumPy -> Tensor (C, H, W)
        # to_tensor scales it back to [0, 1] automatically
        # Replace F.to_tensor(enhanced_np) with:
        return torch.from_numpy(enhanced_np).permute(2, 0, 1).float() / 255.0


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


import torch
import torch.nn.functional as F


class AdaptiveCrop:
    def __init__(self, threshold=0.05, fit_to_square=True):
        """
        threshold: % of active pixels to ignore at edges (denoising).
        fit_to_square: If True, pads the result with black to maintain aspect ratio.
        """
        self.threshold = threshold
        self.fit_to_square = fit_to_square

    def __call__(self, img_tensor):
        # 1. Create a binary mask of active (non-black) pixels
        # Summing across channels; if sum > 0, the pixel is active
        mask = (img_tensor.sum(dim=0) > 0).float()
        total_active = mask.sum()

        if total_active == 0:
            return img_tensor  # Return original if image is completely black

        # 2. Row/Column sums for marginal analysis
        cols = mask.sum(dim=0)  # Shape: [W]
        rows = mask.sum(dim=1)  # Shape: [H]

        def get_limits(sums, total):
            cum_sum = torch.cumsum(sums, dim=0)
            # Find indices where the cumulative active pixels meet the threshold
            start = torch.searchsorted(cum_sum, total * (self.threshold / 2))
            end = torch.searchsorted(cum_sum, total * (1 - self.threshold / 2))
            return start.item(), end.item()

        left, right = get_limits(cols, total_active)
        top, bottom = get_limits(rows, total_active)

        # 3. Perform the initial crop
        cropped = img_tensor[:, top:max(top + 1, bottom), left:max(left + 1, right)]

        # 4. Optional: Pad to Square (Letterboxing)
        if self.fit_to_square:
            c, h, w = cropped.shape
            max_side = max(h, w)

            # Calculate necessary padding to center the animal
            pad_h = (max_side - h) // 2
            pad_w = (max_side - w) // 2

            # Padding order: (left, right, top, bottom)
            padding = (
                pad_w, max_side - w - pad_w,
                pad_h, max_side - h - pad_h
            )

            cropped = F.pad(cropped, padding, mode='constant', value=0)

        return cropped