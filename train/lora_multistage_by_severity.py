"""
train_lora_multistage_by_severity.py

Multi-stage LoRA training script (per severity level) for Stable Diffusion v1.5 + ControlNet.

Pipeline:
- Load a CSV split file containing sat_path, svi_path, and severity labels
- Filter samples by severity (mild / moderate / severe)
- Convert satellite image into polar-transformed conditioning image
- Train LoRA adapters on the UNet while freezing VAE, text encoder, ControlNet, and UNet base weights
- Save one LoRA adapter per severity

Notes:
- CONTROLNET_PATH must point to a folder that contains a valid `config.json`.
- This script expects image files referenced in CSV to exist under IMAGE_DIR (by basename match).
"""

import os
import gc
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    ControlNetModel,
)
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model


# =========================
# 🔧 Absolute Path Settings
# =========================
DATA_ROOT = r"D:\yifan_2025\data"
BASE_OUTPUT_DIR = os.path.join(DATA_ROOT, "output_lora_multistage")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train_split.csv")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")

# Model ID / Path
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# 🔥 Must point to the *converted* checkpoint folder containing `config.json`
CONTROLNET_PATH = os.path.join(
    DATA_ROOT,
    "output_controlnet_sat2street",
    "checkpoint-epoch-20-converted",
)

# Training Hyperparameters
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
EPOCHS_PER_LORA = 10
LEARNING_RATE = 1e-4

# Mapping (based on your CSV label conventions)
SEVERITY_MAPPING = {
    "mild": "0_MinorDamage",
    "moderate": "1_ModerateDamage",
    "severe": "2_SevereDamage",
}


# =========================
# 1) Dataset Definition
# =========================
class SeverityDataset(Dataset):
    def __init__(self, csv_path: str, img_root: str, severity_key: str, size: int = 512):
        self.size = size
        self.root_dir = img_root

        target_label = SEVERITY_MAPPING.get(severity_key)
        print(f"📖 [{severity_key.upper()}] Filtering label: '{target_label}'")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Strip whitespace to ensure exact match
        df["severity_clean"] = df["severity"].astype(str).str.strip()
        self.data = df[df["severity_clean"] == target_label].reset_index(drop=True)

        if len(self.data) == 0:
            print("⚠️ Warning: No matching samples were found.")
        else:
            print(f"✅ Loaded {len(self.data)} samples.")

        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def polar_transform(self, img_pil: Image.Image) -> Image.Image:
        """Convert an RGB PIL image to a polar representation (OpenCV), rotate, and resize."""
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
        center = (w / 2, h / 2)

        polar_img = cv2.linearPolar(
            img_np,
            center,
            w / 2,
            cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
        )
        polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        polar_img = cv2.resize(polar_img, (self.size, self.size))

        return Image.fromarray(polar_img)

    @staticmethod
    def _to_readable_severity(raw_label: str) -> str:
        """Convert '0_MinorDamage' -> 'minor damage' etc. (best-effort)."""
        s = str(raw_label)
        s = s.replace("_", " ")
        s = s.replace("0", "").replace("1", "").replace("2", "")
        return s.strip().lower()

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        sat_name = os.path.basename(str(row["sat_path"]))
        svi_name = os.path.basename(str(row["svi_path"]))

        sat_path = os.path.join(self.root_dir, sat_name)
        svi_path = os.path.join(self.root_dir, svi_name)

        try:
            sat_img = Image.open(sat_path).convert("RGB").resize((self.size, self.size))
            svi_img = Image.open(svi_path).convert("RGB").resize((self.size, self.size))

            # Conditioning image (polar transform of satellite)
            cond_img = self.polar_transform(sat_img)

            # Prompt text
            readable_severity = self._to_readable_severity(row["severity"])
            prompt = f"street view, realistic, {readable_severity}, disaster scene"

            return {
                "pixel_values": self.transform(svi_img),
                "conditioning_pixel_values": self.transform(cond_img),
                "prompt": prompt,
            }
        except Exception:
            # If an image is missing or corrupted, skip it safely.
            return None


def collate_fn(examples):
    examples = [ex for ex in examples if ex is not None]
    if len(examples) == 0:
        return None

    pixel_values = torch.stack([x["pixel_values"] for x in examples])
    condition = torch.stack([x["conditioning_pixel_values"] for x in examples])
    prompts = [x["prompt"] for x in examples]

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": condition,
        "prompts": prompts,
    }


# =========================
# 2) Training Function
# =========================
def train_one_severity(severity_key: str):
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=GRAD_ACCUMULATION,
    )

    save_dir = os.path.join(BASE_OUTPUT_DIR, f"lora_{severity_key}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🚀 [START] Training LoRA for severity: {severity_key.upper()}")

    # 1) Validate ControlNet checkpoint folder
    print(f"🔍 Checking ControlNet path: {CONTROLNET_PATH}")
    config_file = os.path.join(CONTROLNET_PATH, "config.json")
    if not os.path.exists(config_file):
        print(f"❌ FATAL: config.json not found under: {CONTROLNET_PATH}")
        print(
            "💡 Please verify the folder. If you only have the original checkpoint "
            "(e.g., output_controlnet_sat2street/checkpoint-epoch-20), you may need "
            "to convert it properly or ensure a valid config.json exists."
        )
        return

    # 2) Prepare data
    try:
        dataset = SeverityDataset(TRAIN_CSV_PATH, IMAGE_DIR, severity_key)
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return

    if len(dataset) == 0:
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 3) Load models
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")

    # Load ControlNet
    try:
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH)
        print("✅ ControlNet loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load ControlNet: {e}")
        return

    # Freeze base parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    unet.requires_grad_(False)

    # 4) Inject LoRA into UNet
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    # Prepare for distributed / mixed precision
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)
    controlnet.to(accelerator.device, dtype=torch.float16)

    # 5) Training loop
    for epoch in range(EPOCHS_PER_LORA):
        unet.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch in progress_bar:
            if batch is None:
                continue

            with accelerator.accumulate(unet):
                # Encode images into latents
                latents = (
                    vae.encode(batch["pixel_values"].to(dtype=torch.float16))
                    .latent_dist.sample()
                    * vae.config.scaling_factor
                )

                # Tokenize prompts and get text embeddings
                input_ids = tokenizer(
                    batch["prompts"],
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)

                encoder_hidden_states = text_encoder(input_ids)[0]

                # Add diffusion noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ControlNet forward (no grad)
                with torch.no_grad():
                    down_res, mid_res = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=batch["conditioning_pixel_values"].to(dtype=torch.float16),
                        return_dict=False,
                    )

                # UNet noise prediction
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[s.to(dtype=torch.float16) for s in down_res],
                    mid_block_additional_residual=mid_res.to(dtype=torch.float16),
                ).sample

                # MSE loss against true noise
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # 6) Save LoRA adapter
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(save_dir)
    print(f"🎉 Saved LoRA adapter for {severity_key} to: {save_dir}")

    # Cleanup
    del unet, controlnet, vae, text_encoder, optimizer, dataloader, dataset
    accelerator.free_memory()
    del accelerator
    gc.collect()
    torch.cuda.empty_cache()


# =========================
# 3) Main Entrypoint
# =========================
if __name__ == "__main__":
    tasks = ["mild", "moderate", "severe"]
    for t in tasks:
        train_one_severity(t)

    print("\n✅ All training tasks completed successfully!")
