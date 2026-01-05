import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, CLIPTextModel
import gc

# ================= Configuration Area =================
LOCAL_IMAGE_DIR = r"D:\yifan_2025\data\images"

class ExpertDataset(Dataset):
    def __init__(self, csv_path, captions_path, size=512):
        self.size = size
        self.data = pd.read_csv(csv_path)
        
        self.caption_map = {}
        if os.path.exists(captions_path):
            cap_df = pd.read_csv(captions_path)
            # Create a map: filename -> description
            self.caption_map = dict(zip(
                cap_df['sat_path'].apply(lambda x: os.path.basename(str(x))), 
                cap_df['description']
            ))
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sat_filename = os.path.basename(str(row['sat_path']))
        sat_path = os.path.join(LOCAL_IMAGE_DIR, sat_filename)
        
        # Compatibility handling: Find street view image path
        if 'svi_path' in row:
            svi_filename = os.path.basename(str(row['svi_path']))
            svi_path = os.path.join(LOCAL_IMAGE_DIR, svi_filename)
        else:
            svi_path = sat_path 
        
        prompt = self.caption_map.get(sat_filename, "street view, realistic, disaster area")

        try:
            if not os.path.exists(sat_path):
                raise FileNotFoundError
                
            sat_img = Image.open(sat_path).convert("RGB")
            # Polar Transform
            img_np = np.array(sat_img)
            center = (img_np.shape[1]//2, img_np.shape[0]//2)
            polar_img = cv2.linearPolar(img_np, center, img_np.shape[1]//2, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
            polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cond_image = Image.fromarray(polar_img)

            if os.path.exists(svi_path):
                target_image = Image.open(svi_path).convert("RGB")
            else:
                target_image = Image.new("RGB", (self.size, self.size))

        except Exception as e:
            cond_image = Image.new("RGB", (self.size, self.size))
            target_image = Image.new("RGB", (self.size, self.size))

        return {
            "pixel_values": self.transform(target_image),
            "conditioning_pixel_values": self.transform(cond_image),
            "prompt": prompt
        }

# ================= 🚀 Optimized Training Function =================
def train_expert(expert_name, train_csv, output_dir, captions_csv):
    # VRAM Cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # 🔥 Optimization 1: Enable Gradient Accumulation
    # This allows a Batch Size of 1 to achieve the effect of Batch Size 4
    ACCUMULATION_STEPS = 4 
    
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=ACCUMULATION_STEPS
    )

    print(f"\n🚀 Starting training for expert: {expert_name}")
    print(f"   💻 Running device: {accelerator.device} (If cpu, GPU is not being used!)")
    
    if str(accelerator.device) == "cpu":
        print("⚠️⚠️⚠️ Warning: Training on CPU! Please check if PyTorch CUDA version is installed!")

    model_id = "runwayml/stable-diffusion-v1-5"
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    controlnet = ControlNetModel.from_unet(unet)

    # 🔥 Optimization 2: Enable xformers (if installed)
    try:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
        print("   ✅ xformers acceleration enabled")
    except:
        print("   ℹ️ xformers not detected, using standard Attention")

    # Freeze components
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()
    
    # Enable gradient checkpointing (Save VRAM)
    controlnet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

    dataset = ExpertDataset(train_csv, captions_csv)
    
    # 🔥 Optimization 3: Reduce Batch Size to 1, rely on accumulation steps
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0) 

    controlnet, optimizer, train_dataloader = accelerator.prepare(controlnet, optimizer, train_dataloader)
    vae.to(accelerator.device, dtype=torch.float16)
    unet.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    epochs = 20
    global_step = 0
    
    print(f"   🏁 Starting {epochs} epochs of training...")
    
    for epoch in range(epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Ep {epoch+1}/{epochs}", leave=False)
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Encode images to latents
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Encode text prompts
                tokens = tokenizer(
                    batch["prompt"], 
                    max_length=tokenizer.model_max_length, 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                
                encoder_hidden_states = text_encoder(tokens)[0]

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ControlNet Forward
                down_res, mid_res = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=batch["conditioning_pixel_values"].to(dtype=torch.float16),
                    return_dict=False,
                )

                # UNet Forward (Calculate Loss)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[s.to(dtype=torch.float16) for s in down_res],
                    mid_block_additional_residual=mid_res.to(dtype=torch.float16),
                ).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
                
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1
                
    # Save the model
    save_path = os.path.join(output_dir, "checkpoint-epoch-20")
    os.makedirs(save_path, exist_ok=True)
    unwrapped = accelerator.unwrap_model(controlnet)
    unwrapped.save_pretrained(os.path.join(save_path, "controlnet"))
    
    print(f"✅ {expert_name} Training completed! Model saved.")
    
    # Clean up
    del controlnet, unet, vae, text_encoder, optimizer
    torch.cuda.empty_cache()
    gc.collect()

print("✅ Optimized environment ready!")
