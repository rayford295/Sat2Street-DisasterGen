import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from tqdm.auto import tqdm
import cv2
import numpy as np
import subprocess
import gc

# ================= 🔧 Global Configuration (Windows Specific) =================

# 1. Base Working Directory (User Specified)
BASE_DIR = r"D:\yifan_2025\data"

# 2. Input File Paths (Ensure these two CSVs are inside BASE_DIR)
FULL_TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")   # Ground Truth (GT)
GEMINI_PRED_CSV = os.path.join(BASE_DIR, "captions.csv")     # Gemini Predictions

# 3. Output Directory Configuration
MOE_DATA_DIR = os.path.join(BASE_DIR, "moe_processed_data")  # Directory for processed CSVs
OUTPUT_ROOT = os.path.join(BASE_DIR, "moe_checkpoints")      # Directory for trained models

# Automatically create directories
os.makedirs(MOE_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 4. Path to your ControlNet training script
# Assuming your training script is in the same directory as this code, named train_controlnet.py
YOUR_TRAINING_SCRIPT = "train_controlnet.py" 

# 🔥 Core Fix: Include all possible label variations to prevent data loss 🔥
LABEL_MAPPING = {
    # --- Mild (0) ---
    "0_MinorDamage": 0, 
    "no-damage": 0, 
    "Mild": 0, "mild": 0,

    # --- Moderate (1) ---
    "1_ModerateDamage": 1, 
    "minor-damage": 1, "major-damage": 1, 
    "Moderate": 1, "moderate": 1,

    # --- Severe (2) (Fixed previous issue) ---
    "2_SevereDamage": 2, 
    "2_Destroyed": 2, "destroyed": 2, 
    "Severe": 2, "severe": 2,
    "2_MajorDamage": 2 # Some datasets count this as Severe
}

CLASS_NAMES = ["Mild", "Moderate", "Severe"]

# ================= 🛠️ Part 1: Data Engineering =================
def step1_prepare_data():
    print(f"\n[Step 1/3] 🧹 Data Cleaning and Splitting...")
    print(f"   Working Directory: {BASE_DIR}")
    
    if not os.path.exists(FULL_TRAIN_CSV):
        raise FileNotFoundError(f"❌ File not found: {FULL_TRAIN_CSV}\nPlease put train_split.csv in the D:\\yifan_2025\\data directory!")

    df_gt = pd.read_csv(FULL_TRAIN_CSV)
    df_gemini = pd.read_csv(GEMINI_PRED_CSV)
    
    # Unify Key (Filename)
    def get_key(path): return os.path.basename(path)
    df_gt['key'] = df_gt['sat_path'].apply(get_key)
    df_gemini['key'] = df_gemini['sat_path'].apply(get_key)
    
    # Map Labels
    df_gt['gt_label'] = df_gt['severity'].apply(lambda x: LABEL_MAPPING.get(str(x).strip(), -1))
    df_gemini['pred_label'] = df_gemini['severity'].apply(lambda x: LABEL_MAPPING.get(str(x).strip().lower().capitalize(), -1))
    
    # Filter out unrecognized labels (-1)
    df_gt = df_gt[df_gt['gt_label'] != -1]

    # --- Task A: Generate Expert Training Data (Full GT) ---
    print("   📦 Generating Expert Datasets...")
    expert_csvs = {}
    for idx, name in enumerate(CLASS_NAMES):
        sub_df = df_gt[df_gt['gt_label'] == idx]
        save_path = os.path.join(MOE_DATA_DIR, f"expert_{name.lower()}.csv")
        sub_df.to_csv(save_path, index=False)
        expert_csvs[name] = save_path
        print(f"      -> {name} Expert: {len(sub_df)} samples (Saved to {os.path.basename(save_path)})")

    # --- Task B: Generate Router Training Data (Consistency Filtering) ---
    print("   🔍 Generating Router Dataset (Consistent Only)...")
    merged = pd.merge(df_gt, df_gemini, on='key', suffixes=('_gt', '_pred'))
    
    # Keep only samples where GT matches Gemini prediction
    consistent_df = merged[merged['gt_label'] == merged['pred_label']].copy()
    
    # Router needs only image path and numeric label
    router_df = consistent_df[['sat_path_gt', 'gt_label']].rename(columns={'sat_path_gt': 'sat_path', 'gt_label': 'label_idx'})
    print(f"      -> Consistent Samples: {len(router_df)} / {len(merged)} (Conflicting samples removed)")
    
    return router_df, expert_csvs

# ================= 🧠 Part 2: Train Router =================
class SatRouterDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Try loading image. If path in CSV is relative, join with BASE_DIR/images
        # Assuming images are in D:\yifan_2025\data\images
        # If image path is already absolute, row["sat_path"] can be used directly
        img_name = os.path.basename(row["sat_path"])
        img_path = os.path.join(BASE_DIR, "images", img_name) 
        
        # If not found, try using path directly from CSV
        if not os.path.exists(img_path):
             img_path = row["sat_path"]

        try:
            img = Image.open(img_path).convert("RGB")
            # Polar transform (Simulate view seen by Expert)
            img_np = np.array(img)
            center = (img_np.shape[1]//2, img_np.shape[0]//2)
            polar_img = cv2.linearPolar(img_np, center, img_np.shape[1]//2, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
            polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = Image.fromarray(polar_img)
        except Exception:
            # print(f"Warning: Could not read {img_path}, skipping.")
            img = Image.new("RGB", (256, 256)) # Return black image to prevent crash
            
        return self.transform(img), int(row['label_idx'])

def step2_train_router(router_df):
    print("\n[Step 2/3] 🧠 Training Consistent Router...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)
    
    dataset = SatRouterDataset(router_df)
    # Dynamically adjust Batch Size
    batch_size = 16 if len(router_df) > 16 else 4
    if len(router_df) == 0:
        print("❌ Error: 0 consistent samples, cannot train Router! Check if filenames in captions.csv and train_split.csv match.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) # Set to 0 for stability on Windows
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    epochs = 12 
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # Use tqdm to show progress
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    save_path = os.path.join(OUTPUT_ROOT, "router_consistent.pth")
    torch.save(model.state_dict(), save_path)
    print(f"   ✅ Router model saved: {save_path}")
    
    # Clear VRAM
    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()

# ================= 🤖 Part 3: Train 3 Experts =================
def run_controlnet_training(expert_name, csv_path, output_dir):
    print(f"\n   ⚙️ [Simulation Start] Training {expert_name} Expert...")
    print(f"      Input Data: {csv_path}")
    print(f"      Model Output: {output_dir}")
    
    # ⚠️ Uncomment the code below to actually run training ⚠️
    # Assuming your train_controlnet.py accepts --train_csv and --output_dir parameters
    
    # cmd = f"accelerate launch {YOUR_TRAINING_SCRIPT} --train_csv \"{csv_path}\" --output_dir \"{output_dir}\" --num_train_epochs 20"
    # print(f"      Executing Command: {cmd}")
    # try:
    #     subprocess.run(cmd, shell=True, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"      ❌ {expert_name} Training Failed: {e}")

    print(f"   ✅ {expert_name} Expert Training Completed (Simulation).")

def step3_train_experts(expert_csvs):
    print("\n[Step 3/3] 🤖 Batch Training Expert Models...")
    
    for name in CLASS_NAMES:
        csv_path = expert_csvs[name]
        output_dir = os.path.join(OUTPUT_ROOT, f"expert_{name.lower()}")
        
        # Check if data exists
        df = pd.read_csv(csv_path)
        if len(df) < 5:
            print(f"⚠️ Skipping {name} Expert, too little data ({len(df)} samples)")
            continue
            
        run_controlnet_training(name, csv_path, output_dir)

# ================= 🚀 Main Program =================
if __name__ == "__main__":
    print(f"🚀 Starting MoE Automated Training Pipeline (Fixed Version)...")
    print(f"📂 Data Storage Root Directory: {BASE_DIR}")
    
    # 1. Prepare Data
    try:
        router_df, expert_csvs = step1_prepare_data()
        
        # 2. Train Router
        if len(router_df) > 0:
            step2_train_router(router_df)
        else:
            print("⚠️ Warning: No consistent samples, skipping Router training.")
        
        # 3. Train Experts
        step3_train_experts(expert_csvs)
        
        print("\n🎉🎉🎉 Pipeline Completed!")
        
    except Exception as e:
        print(f"\n❌ Program Error: {e}")
        import traceback
        traceback.print_exc()
