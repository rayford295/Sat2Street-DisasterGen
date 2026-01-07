import os
import shutil
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from pytorch_fid import fid_score
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image

# ================= 🔧 Configuration =================
# Root directory of results (must match SAVE_ROOT used during generation)
EVAL_ROOT = r"D:\yifan_2025\evaluation_results"

# Methods to evaluate (folder names)
METHODS = [
    "Pix2Pix",
    "SD1.5_ControlNet",
    "SD1.5_ControlNet_Gemini",
    "MoE_Ours"
]

# Category folders
CATEGORIES = ["Mild", "Moderate", "Severe"]

# Temporary workspace for FID (real/gen split required by pytorch-fid)
TEMP_FID_ROOT = r"D:\yifan_2025\temp_fid_workspace"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 🛠️ Utilities =================

def calculate_pixel_metrics(real_img, gen_img):
    """Compute SSIM and PSNR for a single image pair."""
    # Convert to numpy arrays in [0, 255]
    real_np = np.array(real_img)
    gen_np = np.array(gen_img)
    
    # PSNR
    psnr = psnr_func(real_np, gen_np, data_range=255)
    
    # SSIM (multichannel)
    ssim = ssim_func(real_np, gen_np, channel_axis=2, data_range=255)
    
    return psnr, ssim

def prepare_fid_folders(method, samples):
    """
    Prepare 'real' and 'gen' folders for pytorch-fid.
    'samples' is a list of (real_path, gen_path) pairs.
    """
    real_dir = os.path.join(TEMP_FID_ROOT, method, "real")
    gen_dir = os.path.join(TEMP_FID_ROOT, method, "gen")
    
    # Clean and recreate directories
    if os.path.exists(real_dir): shutil.rmtree(real_dir)
    if os.path.exists(gen_dir): shutil.rmtree(gen_dir)
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    
    print(f"   📂 [FID Prep] Sorting {len(samples)} pairs into temporary folders...")
    
    for idx, (real_path, gen_path) in enumerate(samples):
        # Copy with new names to avoid collisions
        shutil.copy(real_path, os.path.join(real_dir, f"{idx}.png"))
        shutil.copy(gen_path, os.path.join(gen_dir, f"{idx}.png"))
        
    return real_dir, gen_dir

# ================= 🚀 Main Evaluation =================

def run_evaluation():
    print("🚀 Starting full evaluation (SSIM, PSNR, LPIPS, FID)...")
    print(f"   Device: {device}")
    
    # 1) Initialize LPIPS (AlexNet backbone is the standard)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    results = []

    for method in METHODS:
        print(f"\n==========================================")
        print(f"📊 Evaluating method: {method}")
        print(f"==========================================")
        
        method_path = os.path.join(EVAL_ROOT, method)
        if not os.path.exists(method_path):
            print(f"❌ Directory not found: {method_path}. Skipping.")
            continue
            
        # Collect all paired images under this method across categories
        all_pairs = []  # list of (real_path, gen_path)
        
        # Accumulators
        val_psnr = []
        val_ssim = []
        val_lpips = []
        
        # Iterate through Mild, Moderate, Severe
        for cat in CATEGORIES:
            cat_path = os.path.join(method_path, cat)
            if not os.path.exists(cat_path):
                continue
                
            files = os.listdir(cat_path)
            # Find files ending with _real.png
            real_files = [f for f in files if f.endswith("_real.png")]
            
            for rf in real_files:
                base_name = rf.replace("_real.png", "")
                gen_file = base_name + "_gen.png"
                
                real_full = os.path.join(cat_path, rf)
                gen_full = os.path.join(cat_path, gen_file)
                
                if os.path.exists(gen_full):
                    all_pairs.append((real_full, gen_full))
        
        if len(all_pairs) == 0:
            print("⚠️ No paired images found. Skipping.")
            continue
            
        print(f"   🔍 Found {len(all_pairs)} test pairs. Computing metrics...")

        # --- Loop over pairs: SSIM, PSNR, LPIPS ---
        for real_p, gen_p in tqdm(all_pairs, desc=f"   Computing Pixel/LPIPS"):
            # Load images
            real_img = Image.open(real_p).convert("RGB")
            gen_img = Image.open(gen_p).convert("RGB")
            
            # 1) Pixel metrics (CPU)
            p, s = calculate_pixel_metrics(real_img, gen_img)
            val_psnr.append(p)
            val_ssim.append(s)
            
            # 2) LPIPS (GPU): convert to tensors in [-1, 1]
            real_t = torch.from_numpy(np.array(real_img)).permute(2,0,1).float()/127.5 - 1.0
            gen_t = torch.from_numpy(np.array(gen_img)).permute(2,0,1).float()/127.5 - 1.0
            
            real_t = real_t.unsqueeze(0).to(device)
            gen_t = gen_t.unsqueeze(0).to(device)
            
            with torch.no_grad():
                l_dist = loss_fn_alex(real_t, gen_t)
                val_lpips.append(l_dist.item())

        # --- FID ---
        # FID needs two separate folders; stage them first
        fid_real_dir, fid_gen_dir = prepare_fid_folders(method, all_pairs)
        
        print("   🧮 Computing FID (this may take a minute)...")
        try:
            # batch_size=50, dims=2048 (InceptionV3 standard)
            fid_value = fid_score.calculate_fid_given_paths(
                [fid_real_dir, fid_gen_dir],
                batch_size=50,
                device=device,
                dims=2048
            )
        except Exception as e:
            print(f"❌ FID computation failed: {e}")
            fid_value = -1

        # --- Aggregate results ---
        avg_psnr = np.mean(val_psnr)
        avg_ssim = np.mean(val_ssim)
        avg_lpips = np.mean(val_lpips)
        
        print(f"   ✅ {method} results:")
        print(f"      SSIM (↑): {avg_ssim:.4f}")
        print(f"      PSNR (↑): {avg_psnr:.4f}")
        print(f"      LPIPS(↓): {avg_lpips:.4f}")
        print(f"      FID  (↓): {fid_value:.4f}")
        
        results.append({
            "Method": method,
            "SSIM": avg_ssim,
            "PSNR": avg_psnr,
            "LPIPS": avg_lpips,
            "FID": fid_value
        })

    # --- Final report ---
    print("\n\n🏆🏆🏆 Final Evaluation Report 🏆🏆🏆")
    df = pd.DataFrame(results)
    # Pretty print
    print(df.to_string(index=False, formatters={
        'SSIM': '{:.4f}'.format,
        'PSNR': '{:.2f}'.format,
        'LPIPS': '{:.4f}'.format,
        'FID': '{:.2f}'.format
    }))
    
    # Save CSV
    save_csv_path = os.path.join(EVAL_ROOT, "final_metrics.csv")
    df.to_csv(save_csv_path, index=False)
    print(f"\n📄 Metrics saved to: {save_csv_path}")
    
    # Clean up temp files (optional)
    # shutil.rmtree(TEMP_FID_ROOT) 

if __name__ == "__main__":
    run_evaluation()
