import os
import cv2
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from safetensors.torch import load_file

# ================= 🔧 Configuration (Method 3) =================
METHOD_NAME = "SD1.5_ControlNet_Gemini"

# 1) Base paths
BASE_DIR = r"D:\yifan_2025\data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")
SAVE_ROOT = r"D:\yifan_2025\evaluation_results"

# 2) Original training weights
RAW_CHECKPOINT_DIR = r"D:\yifan_2025\data\output_controlnet_gemini_prompt\checkpoint-epoch-20"

# 3) Output directory for the repaired (Diffusers-format) model
FIXED_MODEL_DIR = r"D:\yifan_2025\data\output_controlnet_gemini_prompt\checkpoint-epoch-20-offline-fixed"

# 4) Gemini annotations
CAPTIONS_CSV = os.path.join(BASE_DIR, "captions.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    "0": "Mild", "mild": "Mild", "0_MinorDamage": "Mild", "no-damage": "Mild",
    "1": "Moderate", "moderate": "Moderate", "1_ModerateDamage": "Moderate", "minor-damage": "Moderate", "major-damage": "Moderate",
    "2": "Severe", "severe": "Severe", "2_SevereDamage": "Severe", "2_Destroyed": "Severe", "destroyed": "Severe", "2_MajorDamage": "Severe"
}

# ================= 🛠️ Module 1: Offline model repair (streamlined config) =================
# Note: Removed options like center_input_sample to avoid compatibility errors.
CONTROLNET_CONFIG = {
    "_class_name": "ControlNetModel",
    "_diffusers_version": "0.14.0",
    "act_fn": "silu",
    "attention_head_dim": 8,
    "block_out_channels": [320, 640, 1280, 1280],
    # "center_input_sample": False,  # removed intentionally
    "class_embed_type": None,
    "conditioning_channels": 3,
    "conditioning_embedding_out_channels": [16, 32, 96, 256],
    "controlnet_conditioning_channel_order": "rgb",
    "cross_attention_dim": 768,
    "down_block_types": [
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"
    ],
    "downsample_padding": 1,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "global_pool_conditions": False,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "only_cross_attention": False,
    "upcast_attention": False,
    "use_linear_projection": False
}

def fix_model_offline():
    print(f"🔧 [Step 1] Repairing model format (offline)...")
    
    # Skip if already repaired
    if os.path.exists(os.path.join(FIXED_MODEL_DIR, "config.json")):
        print("✅ Repaired model detected; proceeding.")
        return FIXED_MODEL_DIR

    safetensors_path = os.path.join(RAW_CHECKPOINT_DIR, "model.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"❌ Fatal: original weights not found at {safetensors_path}")
        return None

    # 1) Initialize structure
    print("🏗️ Initializing ControlNet structure...")
    try:
        # Pass the streamlined config via **kwargs
        controlnet = ControlNetModel(**CONTROLNET_CONFIG)
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return None

    # 2) Load weights
    print("📥 Loading safetensors...")
    state_dict = load_file(safetensors_path)
    
    # 3) Fix keys by stripping 'model.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k.replace("model.", "")] = v
        else:
            new_state_dict[k] = v
            
    # 4) Load into model
    m, u = controlnet.load_state_dict(new_state_dict, strict=False)
    print(f"📊 Weights loaded (missing: {len(m)}, unexpected: {len(u)})")

    # 5) Save Diffusers-format model
    print(f"💾 Saving repaired model to: {FIXED_MODEL_DIR}")
    os.makedirs(FIXED_MODEL_DIR, exist_ok=True)
    controlnet.save_pretrained(FIXED_MODEL_DIR)
    
    return FIXED_MODEL_DIR

# ================= 🛠️ Module 2: Data processing & prompt loading =================

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

def load_caption_map():
    if not os.path.exists(CAPTIONS_CSV):
        print(f"⚠️ {CAPTIONS_CSV} not found. Falling back to default prompts.")
        return {}
    
    print(f"📖 Loading Gemini caption library...")
    df = pd.read_csv(CAPTIONS_CSV)
    caption_map = {}
    for _, row in df.iterrows():
        # Key: file name (e.g., 000123.png)
        fname = os.path.basename(str(row['sat_path']))
        desc = str(row['description'])
        if len(desc) > 5:
            caption_map[fname] = desc
    print(f"✅ Indexed {len(caption_map)} captions")
    return caption_map

def save_comparison(method_name, filename, category, real_path, generated_img):
    real_img = Image.open(real_path).convert("RGB")
    original_size = real_img.size
    gen_resized = generated_img.resize(original_size, Image.LANCZOS)
    
    out_dir = os.path.join(SAVE_ROOT, method_name, category)
    os.makedirs(out_dir, exist_ok=True)
    
    real_img.save(os.path.join(out_dir, f"{filename}_real.png"))
    gen_resized.save(os.path.join(out_dir, f"{filename}_gen.png"))

# ================= 🚀 Module 3: Main =================

def run_gemini_inference():
    # 1) Repair model
    model_path = fix_model_offline()
    if not model_path: return

    print(f"\n🚀 [Step 2] Starting generation: {METHOD_NAME}")
    
    # 2) Load captions
    caption_map = load_caption_map()

    # 3) Build pipeline
    try:
        controlnet = ControlNetModel.from_pretrained(model_path, torch_dtype=torch.float16)
        # Attempts to load SD 1.5; uses local cache if available
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=controlnet, 
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(DEVICE)
        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        print("✅ Pipeline loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load SD 1.5: {e}")
        print("💡 Tip: ensure SD 1.5 is cached locally or that the machine can reach Hugging Face.")
        return

    # 4) Read test split
    df = pd.read_csv(TEST_CSV)
    print(f"📄 Test set size: {len(df)}")

    # 5) Generation loop
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        sat_name = os.path.basename(str(row['sat_path']))
        svi_name = os.path.basename(str(row['svi_path']))
        sat_full = os.path.join(IMAGE_DIR, sat_name)
        svi_full = os.path.join(IMAGE_DIR, svi_name)
        fname = os.path.splitext(sat_name)[0]
        
        raw_sev = str(row['severity'])
        category = LABEL_MAP.get(raw_sev, "Moderate")

        if not os.path.exists(sat_full): continue

        # A) Conditioning image
        cond_img = preprocess_satellite(sat_full, resolution=512)
        
        # B) Prompt selection: Gemini caption takes precedence over default
        if sat_name in caption_map:
            gemini_desc = caption_map[sat_name]
            prompt = f"{gemini_desc}, realistic, photorealistic, 8k, highly detailed, street view"
        else:
            prompt = f"street view photography, realistic, ground level view, {raw_sev.replace('_', ' ').lower()}, high quality, 4k"

        # C) Inference
        with torch.no_grad():
            image = pipe(
                prompt, 
                image=cond_img, 
                negative_prompt="low quality, bad quality, sketches, cartoon, blur, distorted, text, watermark",
                num_inference_steps=20, 
                controlnet_conditioning_scale=1.0, 
                guidance_scale=7.5
            ).images[0]
        
        # D) Save results (resized to match the real SVI)
        save_comparison(METHOD_NAME, fname, category, svi_full, image)

    print(f"\n🎉 Completed! Results are saved in: {os.path.join(SAVE_ROOT, METHOD_NAME)}")

if __name__ == "__main__":
    run_gemini_inference()
