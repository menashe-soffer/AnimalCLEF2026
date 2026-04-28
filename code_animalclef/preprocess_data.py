import os
import cv2
import numpy as np
import torch
from pathlib import Path

from gluefactory.settings import DATA_PATH
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

# Import your custom tools from your local file
from image_tools import AdaptiveCrop, UnderwaterEnhance
from paths_and_constants import *


def preprocess_lynx_dataset(input_root, output_root, target_size=(384, 384)):
    input_path = Path(input_root)
    output_path = Path(output_root)

    # 1. Initialize our "Heavy" Logic
    # We use ToTensor at the start so the custom classes work as intended
    cropper = AdaptiveCrop(threshold=0.05, fit_to_square=True)
    enhancer = UnderwaterEnhance(clip_limit=1.5)

    # 2. Gather all image paths (recursive)
    extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in extensions]

    print(f"Found {len(image_files)} images. Starting offline preprocessing...")

    for img_p in tqdm(image_files):
        # Create the corresponding output path
        relative_path = img_p.relative_to(input_path)
        target_p = output_path / relative_path

        # Ensure sub-folders exist
        target_p.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load
            with Image.open(img_p) as img:
                # To maintain consistency with your training pipeline:
                # PIL -> Tensor -> Crop -> Enhance -> Resize -> PIL -> Save
                img_t = TF.to_tensor(img.convert('RGB'))

                # Apply the bottleneck logic once here
                img_t = cropper(img_t)
                img_t = enhancer(img_t)

                # Resize to target
                img_t = TF.resize(img_t, target_size, interpolation=transforms.InterpolationMode.BILINEAR)

                # Convert back to PIL to save
                final_img = TF.to_pil_image(img_t)
                final_img.save(target_p, quality=95)

        except Exception as e:
            print(f"Error processing {img_p}: {e}")


if __name__ == "__main__":
    # Update these paths for your environment
    for name in ['LynxID2025', 'SalamanderID2025', 'SeaTurtleID2022'][1:]:

        IN_DIR = os.path.join(ROOT_DATA, 'images', name)
        OUT_DIR = os.path.join(ROOT_DATA, 'images', '{}_preproc'.format(name))

        preprocess_lynx_dataset(IN_DIR, OUT_DIR)

