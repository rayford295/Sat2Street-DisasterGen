import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import models, transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from safetensors.torch import load_file

# ================= 🔧 Configuration =================
METHOD_NAME = "MoE_Ours"

# 1) Base paths
BASE_DIR = r"D:\yifan_2025\data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")
SAVE_ROOT = r"D:\yifan_2025\evaluation_results"
CAPTIONS_CSV = os.path.join(BASE_DIR, "captions.csv")

# 2) Expert model directories (each should point to checkpoint-epoch-20)
DIR_MILD = r"D:\yifan_2025\data\moe_checkpoints\expert_mild\checkpoint-epoch-20"
DIR_MOD  = r"D:\yifan_2025\data\moe_checkpoints\expert_moderate\checkpoint-epoch-20"
DIR_SEV  = r"D:\yifan_2025\data\moe_checkpoints\expert_severe\checkpoint-epoch-20"

# 3) Router weights
PATH_ROUTER = r"D:\yifan_2025\data\moe_checkpoints\router_consistent.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    "0": "Mild", "mild": "Mild", "0_MinorDamage": "Mild", "no-damage": "Mild",
    "1": "Moderate", "moderate": "Moderate", "1_ModerateDamage": "Moderate", "minor-damage": "Moderate", "major-damage": "Moderate",
    "2": "Severe", "severe": "Severe", "2_SevereDamage": "Severe", "2_Destroyed": "Severe", "destroyed": "Severe", "2_MajorDamage": "Severe"
}

# ================= 🛠️ Smart path resolution (fixed newline issue) =================

def get_valid_model_path(base_dir, expert_name):
    """
    Resolve a valid Diffusers-format model directory for a given expert.
    """
    print(f"🔍 Checking {expert_name} expert path...")
    
    # Case A: Standard format under a 'controlnet' subfolder (produced by your training code)
    #     # ✅ Ensure this line is on the next line, not inline with the comment
    sub_path = os.path.join(base_dir, "controlnet")
    
    if os.path.exists(os.path.join(sub_path, "config.json")):
        print(f"   ✅ Detected standard Diffusers format: {sub_path}")
        return sub_path
        
    # Case B: Standard format directly under the base directory
    if os.path.exists(os.path.join(base_dir, "config.json")):
        print(f"   ✅ Detected standard Diffusers format: {base_dir}")
        return base_dir

    print(f"   ❌ Error: No config.json found in {base_dir}. Please verify that training completed.")
    return None

# ================= 🛠️ Helpers =================

def load_router():
    print("🧠 Loading Router...")
    if not os.path.exists(PATH_ROUTER):
        print(f"❌ Router not found: {PATH_ROUTER}")
        return None
        
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    try:
        model.load_state_dict(torch.load(PATH_ROUTER, map_location=DEVICE))
    except:
        # Handle possible DataParallel checkpoints
        state_dict = torch.load(PATH_ROUTER, map_location=DEVICE)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        
    model.to(DEVICE).eval()
    return model

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
    if not os.path.exists(CAPTIONS_CSV): return {}
    df = pd.read_csv(CAPTIONS_CSV)
    return dict(zip(df['sat_path'].apply(os.path.basename), df['description']))

def save_comparison(method_name, filename, category, real_path, generated_img):
    real_img = Image.open(real_path).convert("RGB")
    gen_resized = generated_img.resize(real_img.size, Image.LANCZOS)
    out_dir = os.path.join(SAVE_ROOT, method_name, category)
    os.makedirs(out_dir, exist_ok=True)
    real_img.save(os.path.join(out_dir, f"{filename}_real.png"))
    gen_resized.save(os.path.join(out_dir, f"{filename}_gen.png"))

# ================= 🚀 Main Inference =================

def run_moe_inference():
    print(f"🚀 Launching MoE joint inference: {METHOD_NAME}")
    
    # 1) Resolve expert model paths
    path_mild = get_valid_model_path(DIR_MILD, "Mild")
    path_mod  = get_valid_model_path(DIR_MOD, "Moderate")
    path_sev  = get_valid_model_path(DIR_SEV, "Severe")
    
    if not all([path_mild, path_mod, path_sev]):
        print("❌ Unable to locate all expert models. Please verify paths.")
        return

    # 2) Load the Router
    router = load_router()
    if not router: return
    
    router_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3) Build pipeline (load all three experts)
    print("🤖 Loading Mixture-of-Experts pipeline...")
    try:
        # Dynamically load three ControlNets
        cnet_mild = ControlNetModel.from_pretrained(path_mild, torch_dtype=torch.float16)
        cnet_mod  = ControlNetModel.from_pretrained(path_mod, torch_dtype=torch.float16)
        cnet_sev  = ControlNetModel.from_pretrained(path_sev, torch_dtype=torch.float16)
        
        # Load SD 1.5 backbone
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[cnet_mild, cnet_mod, cnet_sev], 
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(DEVICE)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        print("✅ Pipeline loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return

    # 4) Data prep
    caption_map = load_caption_map()
    df = pd.read_csv(TEST_CSV)
    print(f"📄 Total samples: {len(df)}")

    # 5) Inference loop
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="MoE Generating"):
        sat_name = os.path.basename(str(row['sat_path']))
        svi_name = os.path.basename(str(row['svi_path']))
        sat_full = os.path.join(IMAGE_DIR, sat_name)
        svi_full = os.path.join(IMAGE_DIR, svi_name)
        
        if not os.path.exists(sat_full): continue

        # A) Conditioning image
        cond_img = preprocess_satellite(sat_full, 512)
        
        # B) Router prediction -> expert weights
        router_in = router_transform(cond_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = router(router_in)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0] 
        
        # C) Prompt (Gemini caption preferred)
        if sat_name in caption_map:
            prompt = f"{caption_map[sat_name]}, realistic, photorealistic, 8k"
        else:
            raw_sev = str(row['severity'])
            prompt = f"street view, realistic, {raw_sev.replace('_', ' ').lower()}, 4k"

        # D) Generation (apply Router weights)
        with torch.no_grad():
            image = pipe(
                prompt,
                image=[cond_img, cond_img, cond_img], 
                negative_prompt="low quality, bad quality, cartoon, blur, text",
                num_inference_steps=20,
                # 🔥 Core: apply router-derived weights to each expert
                controlnet_conditioning_scale=[float(probs[0]), float(probs[1]), float(probs[2])],
                guidance_scale=7.5
            ).images[0]

        # E) Save
        category = LABEL_MAP.get(str(row['severity']), "Moderate")
        save_comparison(METHOD_NAME, os.path.splitext(sat_name)[0], category, svi_full, image)

    print(f"\n🎉 MoE inference complete! Results saved to: {os.path.join(SAVE_ROOT, METHOD_NAME)}")

if __name__ == "__main__":
    run_moe_inference()
