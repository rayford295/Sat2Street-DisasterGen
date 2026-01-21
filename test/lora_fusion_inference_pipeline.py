"""
infer_controlnet_lora_fusion_router.py

LoRA-fusion inference for Stable Diffusion v1.5 + ControlNet (satellite->street-view),
with optional router-based soft gating.

Key features:
- Loads a base ControlNet checkpoint (converted folder with config.json)
- Loads three LoRA adapters (mild / moderate / severe) onto UNet via PEFT
- Computes mixture weights either from:
    (a) a router classifier (ResNet-18), or
    (b) oracle labels from the CSV (severity column)
- Attempts to create a temporary weighted adapter ("mixed") via add_weighted_adapter()
  and set_adapter("mixed"); if unsupported, falls back to hard gating (argmax).
- Generates images and saves side-by-side comparisons (real vs generated) per category.
- Deletes temporary "mixed" adapter per sample to avoid GPU memory blow-up.

Assumptions:
- CSV fields: sat_path, svi_path, severity
- IMAGE_DIR stores images by basename (CSV may store relative/absolute paths; we use basename)
- CAPTIONS_CSV (optional): sat_path basename -> description (caption prompt)
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision import models, transforms
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

# PEFT for adapter loading & management
from peft import PeftModel


# =========================
# 🔧 Configuration
# =========================
METHOD_NAME = "LoRA_Fusion_Ours"

# 1) Paths
BASE_DIR = r"D:\yifan_2025\data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")
SAVE_ROOT = r"D:\yifan_2025\evaluation_results"
CAPTIONS_CSV = os.path.join(BASE_DIR, "captions.csv")

# 2) Model checkpoints
# Base ControlNet (must be the converted folder containing config.json)
PATH_BASE_CONTROLNET = os.path.join(
    BASE_DIR, "output_controlnet_sat2street", "checkpoint-epoch-20-converted"
)

# LoRA adapters (per severity)
PATH_LORA_MILD = os.path.join(BASE_DIR, "output_lora_multistage", "lora_mild")
PATH_LORA_MOD = os.path.join(BASE_DIR, "output_lora_multistage", "lora_moderate")
PATH_LORA_SEV = os.path.join(BASE_DIR, "output_lora_multistage", "lora_severe")

# 3) Router settings
USE_ROUTER = True
PATH_ROUTER = os.path.join(BASE_DIR, "moe_checkpoints", "router_consistent.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    "0": "Mild",
    "mild": "Mild",
    "0_MinorDamage": "Mild",
    "1": "Moderate",
    "moderate": "Moderate",
    "1_ModerateDamage": "Moderate",
    "2": "Severe",
    "severe": "Severe",
    "2_SevereDamage": "Severe",
}


# =========================
# 🛠 Helper Functions
# =========================
def load_router():
    """Load the router classifier (ResNet-18, 3-class)."""
    print("🧠 Loading router...")
    if not os.path.exists(PATH_ROUTER):
        print(f"⚠️ Router checkpoint not found: {PATH_ROUTER}. Switching to Oracle mode.")
        return None

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)

    try:
        state_dict = torch.load(PATH_ROUTER, map_location=DEVICE)
        # Strip DDP "module." prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ Failed to load router: {e}")
        return None

    model.to(DEVICE).eval()
    return model


def preprocess_satellite(sat_path: str, resolution: int = 512) -> Image.Image:
    """Convert satellite image into a polar-transformed conditioning image."""
    sat_img = Image.open(sat_path).convert("RGB")
    sat_img = sat_img.resize((resolution, resolution), Image.BICUBIC)

    img_np = np.array(sat_img)
    h, w = img_np.shape[:2]
    center = (w / 2, h / 2)

    polar_img = cv2.linearPolar(
        img_np,
        center,
        w / 2,
        cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    polar_img = cv2.resize(polar_img, (resolution, resolution))
    return Image.fromarray(polar_img)


def load_caption_map():
    """Load optional caption prompts mapping: basename(sat_path) -> description."""
    if not os.path.exists(CAPTIONS_CSV):
        return {}
    df = pd.read_csv(CAPTIONS_CSV)
    return dict(zip(df["sat_path"].apply(os.path.basename), df["description"]))


def save_comparison(method_name: str, filename: str, category: str, real_path: str, generated_img: Image.Image):
    """Save real and generated images (aligned to the real image size)."""
    real_img = Image.open(real_path).convert("RGB")
    gen_resized = generated_img.resize(real_img.size, Image.LANCZOS)

    out_dir = os.path.join(SAVE_ROOT, method_name, category)
    os.makedirs(out_dir, exist_ok=True)

    real_img.save(os.path.join(out_dir, f"{filename}_real.png"))
    gen_resized.save(os.path.join(out_dir, f"{filename}_gen.png"))


def get_oracle_weights(label_str):
    """Oracle weights from ground-truth severity label."""
    category = LABEL_MAP.get(str(label_str), "Moderate")
    if category == "Mild":
        return [1.0, 0.0, 0.0]
    if category == "Moderate":
        return [0.0, 1.0, 0.0]
    if category == "Severe":
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


def validate_controlnet_folder(controlnet_path: str) -> bool:
    """Ensure the ControlNet folder has a config.json (diffusers format)."""
    cfg = os.path.join(controlnet_path, "config.json")
    if not os.path.exists(cfg):
        print(f"❌ FATAL: config.json not found under: {controlnet_path}")
        return False
    return True


# =========================
# 🚀 Main Inference
# =========================
def run_lora_fusion_inference():
    print(f"🚀 Starting LoRA Fusion inference: {METHOD_NAME}")
    os.makedirs(os.path.join(SAVE_ROOT, METHOD_NAME), exist_ok=True)

    # 0) Sanity checks
    if not os.path.exists(TEST_CSV):
        print(f"❌ TEST_CSV not found: {TEST_CSV}")
        return
    if not validate_controlnet_folder(PATH_BASE_CONTROLNET):
        return

    # 1) Router (optional)
    router = None
    router_transform = None
    if USE_ROUTER:
        router = load_router()
        if router is not None:
            router_transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            )

    # 2) Load SD1.5 + ControlNet pipeline
    print("🤖 Loading Stable Diffusion v1.5 + base ControlNet...")
    try:
        controlnet = ControlNetModel.from_pretrained(
            PATH_BASE_CONTROLNET, torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(DEVICE)

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception as e:
        print(f"❌ Failed to load base pipeline: {e}")
        return

    # 3) Load LoRA adapters via PEFT
    print("🧬 Attaching LoRA adapters via PEFT...")
    try:
        base_unet = pipe.unet

        # Wrap UNet with PEFT and load the first adapter (mild)
        peft_unet = PeftModel.from_pretrained(
            base_unet, PATH_LORA_MILD, adapter_name="mild"
        )

        # Load additional adapters
        peft_unet.load_adapter(PATH_LORA_MOD, adapter_name="mod")
        peft_unet.load_adapter(PATH_LORA_SEV, adapter_name="sev")

        # Put back into pipeline
        pipe.unet = peft_unet

        print(f"✅ LoRA loaded. Available adapters: {list(pipe.unet.peft_config.keys())}")
    except Exception as e:
        print(f"❌ FATAL: LoRA loading error: {e}")
        print("Please verify output_lora_multistage and its subfolders are complete.")
        return

    # 4) Load data
    caption_map = load_caption_map()
    df = pd.read_csv(TEST_CSV)
    print(f"📄 Total samples: {len(df)}")

    # 5) Inference loop
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LoRA Fusion Generating"):
        sat_name = os.path.basename(str(row["sat_path"]))
        svi_name = os.path.basename(str(row["svi_path"]))

        sat_full = os.path.join(IMAGE_DIR, sat_name)
        svi_full = os.path.join(IMAGE_DIR, svi_name)

        if not os.path.exists(sat_full):
            continue

        # A) Conditioning image
        cond_img = preprocess_satellite(sat_full, resolution=512)

        # B) Router weights or oracle weights
        if router is not None and router_transform is not None:
            router_in = router_transform(cond_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = router(router_in)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            weights = [float(probs[0]), float(probs[1]), float(probs[2])]
        else:
            weights = get_oracle_weights(row["severity"])

        # C) Prompt
        if sat_name in caption_map:
            prompt = f"{caption_map[sat_name]}, realistic, photorealistic, 8k"
        else:
            raw_sev = str(row["severity"]).replace("_", " ").lower()
            prompt = f"street view, realistic, {raw_sev}, 4k"

        # D) Apply weighted adapter fusion if supported; otherwise hard gating
        try:
            # Create a temporary fused adapter called "mixed"
            pipe.unet.add_weighted_adapter(
                adapters=["mild", "mod", "sev"],
                weights=weights,
                adapter_name="mixed",
                combination_type="linear",
            )
            pipe.unet.set_adapter("mixed")
        except Exception:
            # Fallback: activate the adapter with max weight (hard gating)
            max_idx = int(np.argmax(weights))
            target = ["mild", "mod", "sev"][max_idx]
            pipe.unet.set_adapter(target)

        # E) Generate image
        with torch.no_grad():
            image = pipe(
                prompt,
                image=cond_img,
                negative_prompt="low quality, bad quality, cartoon, blur, text, watermark",
                num_inference_steps=20,
                guidance_scale=7.5,
            ).images[0]

        # Cleanup temporary adapter to avoid memory / config growth
        if "mixed" in getattr(pipe.unet, "peft_config", {}):
            try:
                pipe.unet.delete_adapter("mixed")
            except Exception:
                pass

        # F) Save comparison
        category = LABEL_MAP.get(str(row["severity"]), "Moderate")
        filename = os.path.splitext(sat_name)[0]
        if os.path.exists(svi_full):
            save_comparison(METHOD_NAME, filename, category, svi_full, image)

    print(f"\n🎉 Done! Results saved to: {os.path.join(SAVE_ROOT, METHOD_NAME)}")


if __name__ == "__main__":
    run_lora_fusion_inference()
