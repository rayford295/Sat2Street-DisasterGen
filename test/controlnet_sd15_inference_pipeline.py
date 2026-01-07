import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel
from safetensors.torch import load_file

# ================= 🔧 Path Configuration =================
METHOD_NAME = "SD1.5_ControlNet"
BASE_DIR = r"D:\yifan_2025\data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")
SAVE_ROOT = r"D:\yifan_2025\evaluation_results"

# Original training checkpoint (Accelerate format)
RAW_CHECKPOINT_DIR = r"D:\yifan_2025\data\output_controlnet_sat2street\checkpoint-epoch-20"
# Destination for the repaired model (Diffusers format)
FIXED_MODEL_DIR = r"D:\yifan_2025\data\output_controlnet_sat2street\checkpoint-epoch-20-converted"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    "0": "Mild", "mild": "Mild", "0_MinorDamage": "Mild", "no-damage": "Mild",
    "1": "Moderate", "moderate": "Moderate", "1_ModerateDamage": "Moderate", "minor-damage": "Moderate", "major-damage": "Moderate",
    "2": "Severe", "severe": "Severe", "2_SevereDamage": "Severe", "2_Destroyed": "Severe", "destroyed": "Severe", "2_MajorDamage": "Severe"
}

# ================= 🛠️ Module 1: Auto-fix model format =================
def fix_model_format():
    """
    Verify Diffusers format (config.json). If missing, convert from safetensors and save.
    """
    print(f"🔧 [Step 1] Checking model format...")

    # If already converted, skip
    if os.path.exists(os.path.join(FIXED_MODEL_DIR, "config.json")):
        print("✅ Repaired model detected (config.json exists). Skipping conversion.")
        return FIXED_MODEL_DIR
    
    print("⚠️ Diffusers-format model not found. Converting from training checkpoint...")

    safetensors_path = os.path.join(RAW_CHECKPOINT_DIR, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"❌ Fatal: original training weights not found. Check path: {safetensors_path}")

    print(f"📂 Loading original weights: {safetensors_path}")
    
    # 1) Initialize architecture (reuse SD1.5 UNet config)
    print("🏗️ Initializing ControlNet structure...")
    try:
        # Requires network access to fetch SD1.5 config
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        controlnet = ControlNetModel.from_unet(unet)
    except Exception as e:
        print(f"❌ Network error: unable to fetch config from Hugging Face: {e}")
        return None

    # 2) Load weights from safetensors and strip the 'model.' prefix if present
    state_dict = load_file(safetensors_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "")] = v
        else:
            new_state_dict[k] = v
            
    # 3) Load into ControlNet and save in Diffusers format
    missing, unexpected = controlnet.load_state_dict(new_state_dict, strict=False)
    print(f"📥 Weights loaded (missing keys: {len(missing)}, unexpected keys: {len(unexpected)})")
    
    os.makedirs(FIXED_MODEL_DIR, exist_ok=True)
    controlnet.save_pretrained(FIXED_MODEL_DIR)
    print(f"💾 Model repaired and saved to: {FIXED_MODEL_DIR}")
    
    return FIXED_MODEL_DIR

# ================= 🛠️ Module 2: Image utilities =================
def preprocess_satellite(sat_path, resolution=512):
    sat_img = Image.open(sat_path).convert("RGB")
    sat_img = sat_img.resize((resolution, resolution), Image.BICUBIC)
    
    img_np = np.array(sat_img)
    h, w = img_np.shape[:2]
    center = (w / 2, h / 2)
    max_radius = w / 2 
    
    polar_img = cv2.linearPolar(img_np, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    polar_img = cv2.resize(polar_img, (resolution, resolution))
    
    return Image.fromarray(polar_img)

def save_comparison(method_name, filename, category, real_path, generated_img):
    real_img = Image.open(real_path).convert("RGB")
    original_size = real_img.size
    gen_resized = generated_img.resize(original_size, Image.LANCZOS)
    
    out_dir = os.path.join(SAVE_ROOT, method_name, category)
    os.makedirs(out_dir, exist_ok=True)
    
    real_img.save(os.path.join(out_dir, f"{filename}_real.png"))
    gen_resized.save(os.path.join(out_dir, f"{filename}_gen.png"))

# ================= 🚀 Module 3: Inference main =================
def run_main():
    # 1) Fix/convert model if needed
    model_path = fix_model_format()
    if not model_path:
        return

    print(f"\n🚀 [Step 2] Starting generation: {METHOD_NAME}")
    
    # 2) Load the repaired ControlNet + SD1.5 pipeline
    try:
        controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=controlnet, 
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(DEVICE)
        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload() 
        print("✅ Inference pipeline loaded successfully!")
    except Exception as e:
        print(f"❌ Pipeline load failed: {e}")
        return

    # 3) Read test CSV
    df = pd.read_csv(TEST_CSV)
    print(f"📄 Test set size: {len(df)} samples")

    # 4) Generate loop
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        sat_name = os.path.basename(str(row['sat_path']))
        svi_name = os.path.basename(str(row['svi_path']))
        sat_full = os.path.join(IMAGE_DIR, sat_name)
        svi_full = os.path.join(IMAGE_DIR, svi_name)
        fname = os.path.splitext(sat_name)[0]
        
        raw_sev = str(row['severity'])
        category = LABEL_MAP.get(raw_sev, "Moderate")

        if not os.path.exists(sat_full): 
            continue

        cond_img = preprocess_satellite(sat_full, resolution=512)
        prompt = f"street view photography, realistic, ground level view, {raw_sev.replace('_', ' ').lower()}, high quality, 4k"
        
        with torch.no_grad():
            image = pipe(
                prompt, image=cond_img, 
                negative_prompt="low quality, bad quality, sketches, cartoon, blur, distorted, text, watermark",
                num_inference_steps=20, controlnet_conditioning_scale=1.0, guidance_scale=7.5
            ).images[0]
        
        save_comparison(METHOD_NAME, fname, category, svi_full, image)

    print(f"\n🎉 Done! Results saved to: {os.path.join(SAVE_ROOT, METHOD_NAME)}")

if __name__ == "__main__":
    run_main()
