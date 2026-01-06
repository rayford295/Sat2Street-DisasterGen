import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
from torchvision import transforms

# ================= 🔧 Global Configuration =================
BASE_DIR = r"D:\yifan_2025\data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")
SAVE_ROOT = r"D:\yifan_2025\evaluation_results"  # Root directory for final results

# Label mapping dictionary
LABEL_MAP = {
    "0": "Mild", "mild": "Mild", "0_MinorDamage": "Mild", "no-damage": "Mild",
    "1": "Moderate", "moderate": "Moderate", "1_ModerateDamage": "Moderate",
    "minor-damage": "Moderate", "major-damage": "Moderate",
    "2": "Severe", "severe": "Severe", "2_SevereDamage": "Severe",
    "2_Destroyed": "Severe", "destroyed": "Severe", "2_MajorDamage": "Severe"
}

# ================= 🛠️ Utility Functions =================

def get_test_data():
    """Load and clean test dataset paths."""
    df = pd.read_csv(TEST_CSV)
    test_samples = []
    
    for _, row in df.iterrows():
        # Extract file names (ignore absolute prefixes in CSV)
        sat_name = os.path.basename(str(row['sat_path']))
        svi_name = os.path.basename(str(row['svi_path']))
        
        # Reconstruct full local paths
        sat_full = os.path.join(IMAGE_DIR, sat_name)
        svi_full = os.path.join(IMAGE_DIR, svi_name)
        
        # Parse severity label
        raw_label = str(row['severity']).split('_')[0]
        category = LABEL_MAP.get(str(row['severity']), None)
        if not category:
            category = LABEL_MAP.get(raw_label, "Moderate")  # Default fallback
            
        if os.path.exists(sat_full) and os.path.exists(svi_full):
            test_samples.append({
                "sat_path": sat_full,
                "svi_path": svi_full,
                "filename": sat_name.split('.')[0],
                "category": category
            })
            
    print(f"✅ Loaded test samples: {len(test_samples)} images")
    return test_samples


def save_comparison(method_name, sample, generated_img):
    """
    Save visual comparison results:
    1. Restore original image size.
    2. Create category-specific folders.
    3. Save real and generated images.
    """
    # 1. Retrieve real image size
    real_img = Image.open(sample['svi_path']).convert("RGB")
    original_size = real_img.size  # (W, H)
    
    # 2. Resize generated image back to original dimensions
    gen_resized = generated_img.resize(original_size, Image.LANCZOS)
    
    # 3. Define output directory structure: D:\...\Method\Category\
    out_dir = os.path.join(SAVE_ROOT, method_name, sample['category'])
    os.makedirs(out_dir, exist_ok=True)
    
    # 4. Save both real and generated images
    real_save_path = os.path.join(out_dir, f"{sample['filename']}_real.png")
    gen_save_path = os.path.join(out_dir, f"{sample['filename']}_gen.png")
    
    real_img.save(real_save_path)
    gen_resized.save(gen_save_path)


def preprocess_satellite(sat_path, size=512):
    """Perform polar coordinate transformation on a satellite image."""
    sat_img = Image.open(sat_path).convert("RGB")
    img_np = np.array(sat_img)
    center = (img_np.shape[1] // 2, img_np.shape[0] // 2)
    polar_img = cv2.linearPolar(
        img_np, center, img_np.shape[1] // 2,
        cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    )
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return Image.fromarray(polar_img).resize((size, size))


print("✅ Environment setup complete! Proceed to run the generation code below.")
