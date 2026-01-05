import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 0. GPU Environment Check
# ==========================================
print(f"🔍 Checking runtime environment...")
if torch.cuda.is_available():
    print(f"✅ GPU detected and ready: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("❌ Fatal error: No GPU detected!")

# ==========================================
# 1. Configuration (please verify paths)
# ==========================================
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5

# 🔥 Output directory
OUTPUT_DIR = r"D:\yifan_2025\data\output_controlnet_gemini_prompt"  # renamed to distinguish experiments
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# File name configuration
TRAIN_CSV = "train_split.csv"
TEST_CSV = "test_split.csv"
CAPTIONS_CSV = "captions.csv"  # 🔥 New: Gemini annotation file

# ==========================================
# 2. Dataset Class (Core Modification Area)
# ==========================================
class Sat2StreetDataset(Dataset):
    def __init__(self, root_dir="./", resolution=512, split="train"):
        self.resolution = resolution
        self.root_dir = root_dir
        
        # 1. Load split CSV
        if split == "train":
            target_csv = TRAIN_CSV
        else:
            target_csv = TEST_CSV
        
        split_path = os.path.join(root_dir, target_csv)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"❌ Split file not found: {split_path}")
        self.df = pd.read_csv(split_path)
        
        # 2. 🔥 Load Gemini annotation file (build lookup dictionary)
        captions_path = os.path.join(root_dir, CAPTIONS_CSV)
        self.caption_map = {}  # Key: filename, Value: description
        
        if os.path.exists(captions_path):
            print(f"📖 [{split.upper()}] Loading Gemini annotations: {captions_path} ...")
            cap_df = pd.read_csv(captions_path)
            # Build mapping: filename -> description
            # Assumes captions.csv contains 'sat_path' and 'description' columns
            for _, row in cap_df.iterrows():
                fname = os.path.basename(row['sat_path'])  # use filename only
                desc = row['description']
                if isinstance(desc, str) and len(desc) > 5:  # simple validity filter
                    self.caption_map[fname] = desc
            print(f"✅ Indexed {len(self.caption_map)} Gemini descriptions")
        else:
            print(f"⚠️ Warning: {captions_path} not found, fallback prompts will be used!")

        print(f"✅ [{split.upper()}] Dataset loaded: {len(self.df)} images")

    def __len__(self):
        return len(self.df)

    def polar_transform(self, img_pil):
        img_np = np.array(img_pil)
        h, w = img_np.shape[:2]
        center = (w / 2, h / 2)
        max_radius = w / 2
        polar_img = cv2.linearPolar(
            img_np, center, max_radius,
            cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
        )
        polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        polar_img = cv2.resize(polar_img, (self.resolution, self.resolution))
        return Image.fromarray(polar_img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sat_name = os.path.basename(row["sat_path"])
        svi_name = os.path.basename(row["svi_path"])
        
        sat_path = os.path.join(self.root_dir, "images", sat_name)
        svi_path = os.path.join(self.root_dir, "images", svi_name)

        # --- Image loading and preprocessing ---
        try:
            sat = Image.open(sat_path).convert("RGB")
            svi = Image.open(svi_path).convert("RGB")
            sat = sat.resize((self.resolution, self.resolution), Image.BICUBIC)
            svi = svi.resize((self.resolution, self.resolution), Image.BICUBIC)
            sat = self.polar_transform(sat)
        except Exception as e:
            print(f"⚠️ Error loading {sat_path}: {e}")
            sat = Image.new('RGB', (self.resolution, self.resolution))
            svi = Image.new('RGB', (self.resolution, self.resolution))

        svi_t = torch.from_numpy(
            np.array(svi).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1)
        sat_t = torch.from_numpy(
            np.array(sat).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        # --- 🔥 Prompt strategy update 🔥 ---
        # 1. Try to retrieve Gemini description
        if sat_name in self.caption_map:
            gemini_desc = self.caption_map[sat_name]
            # Append quality/style tokens for Stable Diffusion consistency
            prompt = (
                f"{gemini_desc}, realistic, photorealistic, 8k, "
                f"highly detailed, street view"
            )
        else:
            # 2. Fallback to original template
            severity = row['severity'] if 'severity' in row else "disaster"
            prompt = (
                f"street view photography, realistic, ground level view, "
                f"{str(severity).replace('_', ' ').lower()}, high quality, 4k"
            )

        return {
            "pixel_values": svi_t,
            "condition_pixel_values": sat_t,
            "input_ids": prompt
        }

# ==========================================
# 3. Helper Function: Validation Visualization
# ==========================================
def log_validation(
    vae, text_encoder, tokenizer, unet, controlnet,
    args, accelerator, weight_dtype, epoch, val_sample
):
    print(f"\n🎨 [Epoch {epoch}] Generating validation image...")
    print(f"📝 Prompt: {val_sample['prompt'][:100]}...")

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=accelerator.unwrap_model(controlnet),
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(accelerator.device)

    validation_image = val_sample["image"]
    validation_prompt = val_sample["prompt"]

    with torch.autocast("cuda"):
        image = pipeline(
            validation_prompt,
            image=validation_image,
            num_inference_steps=20,
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]

    save_path = os.path.join(OUTPUT_DIR, f"val_epoch_{epoch+1}.png")
    w, h = validation_image.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(validation_image, (0, 0))
    combined.paste(image, (w, 0))
    combined.save(save_path)
    print(f"🖼️ Validation image saved to: {save_path}")
    
    del pipeline
    torch.cuda.empty_cache()

# ==========================================
# 4. Training Entry Point
# ==========================================
def train_main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    accelerator = Accelerator(mixed_precision="fp16")
    
    # --- Prepare validation sample (Gemini prompt supported) ---
    print("🧊 Locking validation sample...")
    test_df = pd.read_csv(os.path.join("./", TEST_CSV))
    first_row = test_df.iloc[0]
    sat_fname = os.path.basename(first_row["sat_path"])
    
    sat_path = os.path.join("./images", sat_fname)
    sat_pil = Image.open(sat_path).convert("RGB").resize((512, 512))
    img_np = np.array(sat_pil)
    center = (256, 256)
    polar_img = cv2.linearPolar(
        img_np, center, 256,
        cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    )
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    val_image_pil = Image.fromarray(polar_img)
    
    val_prompt = "street view photography, realistic, high quality"
    captions_path = os.path.join("./", CAPTIONS_CSV)
    if os.path.exists(captions_path):
        cap_df = pd.read_csv(captions_path)
        match = cap_df[
            cap_df['sat_path'].apply(os.path.basename) == sat_fname
        ]
        if not match.empty:
            gemini_desc = match.iloc[0]['description']
            val_prompt = (
                f"{gemini_desc}, realistic, photorealistic, 8k, "
                f"highly detailed, street view"
            )
            print("✅ Validation image matched with Gemini description.")
        else:
            print("⚠️ No Gemini description found for validation image.")

    val_sample = {"image": val_image_pil, "prompt": val_prompt}

    # 1. Load dataset
    train_dataset = Sat2StreetDataset(
        root_dir="./", split="train", resolution=512
    )
    
    # 2. Load models
    print("🚀 Initializing ControlNet & Stable Diffusion 1.5...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)

    # 3. Freeze parameters
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    optimizer = torch.optim.AdamW(
        controlnet.parameters(), lr=LEARNING_RATE
    )

    def collate_fn(examples):
        pixel_values = torch.stack([x["pixel_values"] for x in examples])
        condition = torch.stack([x["condition_pixel_values"] for x in examples])
        prompts = [x["input_ids"] for x in examples]
        inputs = tokenizer(
            prompts,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "pixel_values": pixel_values,
            "condition_pixel_values": condition,
            "input_ids": inputs.input_ids
        }

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=0
    )
    controlnet, optimizer, train_dataloader = accelerator.prepare(
        controlnet, optimizer, train_dataloader
    )
    
    vae.to(accelerator.device, dtype=torch.float16)
    unet.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    print("🔥 Training started! Mode: Semantic Injection")

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                latents = (
                    vae.encode(batch["pixel_values"].to(dtype=torch.float16))
                    .latent_dist.sample() * 0.18215
                )
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps
                )
                encoder_hidden_states = text_encoder(
                    batch["input_ids"]
                )[0]
                
                down, mid = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=batch["condition_pixel_values"].to(
                        dtype=torch.float16
                    ),
                    return_dict=False,
                )
                
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        s.to(dtype=torch.float16) for s in down
                    ],
                    mid_block_additional_residual=mid.to(
                        dtype=torch.float16
                    ),
                ).sample
                
                loss = F.mse_loss(
                    noise_pred.float(),
                    noise.float(),
                    reduction="mean"
                )
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            
        save_path = os.path.join(
            OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}"
        )
        accelerator.save_state(save_path)
        
        # Validation
        log_validation(
            vae, text_encoder, tokenizer, unet, controlnet,
            None, accelerator, torch.float16, epoch, val_sample
        )

if __name__ == "__main__":
    train_main()
