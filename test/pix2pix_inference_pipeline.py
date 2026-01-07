import torch
import torch.nn as nn
from torchvision import transforms
from tqdm.auto import tqdm
import os
import cv2
import numpy as np
from PIL import Image

# ================= 🔧 Pix2Pix Configuration =================
METHOD_NAME = "Pix2Pix"
# Path to the generator weights you just exported
MODEL_PATH = r"D:\yifan_2025\data\output_pix2pix\generator_epoch_100.pth" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 🏗️ The model architecture must match training exactly =================
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize: layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )
    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2); d4 = self.down4(d3)
        d5 = self.down5(d4); d6 = self.down6(d5); d7 = self.down7(d6); d8 = self.down8(d7)
        u1 = self.up1(d8, d7); u2 = self.up2(u1, d6); u3 = self.up3(u2, d5); u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3); u6 = self.up6(u5, d2); u7 = self.up7(u6, d1)
        return self.final(u7)

# ================= 🚀 Run generation =================

def run_pix2pix():
    print(f"🚀 Start generation: {METHOD_NAME}")
    
    # 1) Instantiate the exact same model used during training
    netG = GeneratorUNet().to(DEVICE)
    
    # 2) Load weights
    try:
        # Use map_location to support both CPU and GPU loading
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        netG.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return

    netG.eval()
    
    # 3) Preprocessing (must be identical to your training polar_transform)
    def preprocess_for_pix2pix(sat_path, size=256):
        # Training first resized, then applied Polar transform; keep the same order
        sat_img = Image.open(sat_path).convert("RGB")
        sat_img = sat_img.resize((size, size), Image.BICUBIC)  # resize first as in training
        
        img_np = np.array(sat_img)
        h, w = img_np.shape[:2]
        center = (w / 2, h / 2)
        max_radius = w / 2 
        # Polar transform
        polar_img = cv2.linearPolar(img_np, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        polar_img = cv2.resize(polar_img, (size, size))  # ensure final size
        
        # Normalize (training used /127.5 - 1.0)
        img_tensor = torch.from_numpy(polar_img.astype(np.float32)).permute(2, 0, 1) / 127.5 - 1.0
        return img_tensor.unsqueeze(0)  # add batch dim

    # Fetch test data
    samples = get_test_data() 
    
    for sample in tqdm(samples, desc="Pix2Pix Inferencing"):
        # Preprocess
        input_tensor = preprocess_for_pix2pix(sample['sat_path'], size=256).to(DEVICE)
        
        with torch.no_grad():
            fake_img = netG(input_tensor)
        
        # Postprocess (de-normalize)
        fake_img = fake_img.squeeze().cpu().detach()
        fake_img = (fake_img + 1) / 2.0  # [-1, 1] -> [0, 1]
        fake_img = torch.clamp(fake_img, 0, 1)
        fake_pil = transforms.ToPILImage()(fake_img)
        
        # Save (automatically resized back to the original SVI size)
        save_comparison(METHOD_NAME, sample, fake_pil)

# Execute
run_pix2pix()
