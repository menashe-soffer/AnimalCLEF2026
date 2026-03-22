import cv2
import numpy as np
from PIL import Image



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
