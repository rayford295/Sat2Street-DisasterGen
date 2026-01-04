import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torchvision.utils import save_image

# ==========================================
# 1. Core Configuration (Config)
# ==========================================
CONFIG = {
    "n_epochs": 100,
    "batch_size": 4,            
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "img_size": 256,
    "lambda_pixel": 100,        
    "output_dir": "./output_pix2pix_polar_fixed", # Modified output directory to prevent overwriting
    "root_dir": "./",           # Assuming Notebook is located inside the data folder
    "train_csv": "train_split.csv", # Your training set file
    "test_csv": "test_split.csv"    # Your testing set file
}

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Dataset Class (Modified: Reads Fixed Split CSV)
# ==========================================
class Sat2StreetDataset(Dataset):
    def __init__(self, root_dir="./", split="train", resolution=256):
        self.resolution = resolution
        self.root_dir = root_dir
        
        # 🔥 Modification: Select different CSV based on split 🔥
        if split == "train":
            csv_filename = CONFIG["train_csv"]
        else:
            csv_filename = CONFIG["test_csv"]
            
        csv_path = os.path.join(root_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Cannot find {csv_path}! Please check if the filename is correct.")

        # Read CSV
        self.df = pd.read_csv(csv_path)
        print(f"✅ [{split.upper()}] Loaded {len(self.df)} images from {csv_filename}")

    def __len__(self):
        return len(self.df)

    # Polar Transformation (Remains unchanged)
    def polar_transform(self, img_pil):
        img_np = np.array(img_pil) 
        h, w = img_np.shape[:2]
        center = (w / 2, h / 2)
        max_radius = w / 2 
        # LinearPolar expansion
        polar_img = cv2.linearPolar(img_np, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        # Rotate 90 degrees to align
        polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        polar_img = cv2.resize(polar_img, (self.resolution, self.resolution))
        return Image.fromarray(polar_img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Column names must match your CSV, assuming 'sat_path' and 'svi_path'
        # Use basename to ensure only the filename is taken, avoiding incorrect absolute paths in CSV
        sat_name = os.path.basename(row["sat_path"])
        svi_name = os.path.basename(row["svi_path"])
        
        # Construct image paths: ./images/xxxx.jpg
        sat_path = os.path.join(self.root_dir, "images", sat_name)
        svi_path = os.path.join(self.root_dir, "images", svi_name)

        try:
            sat = Image.open(sat_path).convert("RGB")
            svi = Image.open(svi_path).convert("RGB")
            
            # Resize
            sat = sat.resize((self.resolution, self.resolution), Image.BICUBIC)
            svi = svi.resize((self.resolution, self.resolution), Image.BICUBIC)
            
            # 🔥 Apply Polar Transformation 🔥
            sat = self.polar_transform(sat)

        except Exception as e:
            # If loading fails, generate a black image to prevent crash
            print(f"⚠️ Error loading: {sat_path} or {svi_path} ({e})")
            sat = Image.new('RGB', (self.resolution, self.resolution))
            svi = Image.new('RGB', (self.resolution, self.resolution))

        # Normalize
        sat_t = torch.from_numpy(np.array(sat).astype(np.float32)).permute(2, 0, 1) / 127.5 - 1.0
        svi_t = torch.from_numpy(np.array(svi).astype(np.float32)).permute(2, 0, 1) / 127.5 - 1.0

        return {"A": sat_t, "B": svi_t}

# ==========================================
# 3. Model Definition (Pix2Pix U-Net & Discriminator)
# ==========================================
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

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

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization: layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# ==========================================
# 4. Training Loop
# ==========================================
if __name__ == "__main__":
    print("🚀 Initializing model (Fixed Splits + Polar Transform)...")
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=CONFIG["lr"], betas=(CONFIG["b1"], CONFIG["b2"]))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=CONFIG["lr"], betas=(CONFIG["b1"], CONFIG["b2"]))
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # 🔥 Load Dataset (Specify split) 🔥
    train_dataset = Sat2StreetDataset(root_dir=CONFIG["root_dir"], split="train", resolution=CONFIG["img_size"])
    val_dataset = Sat2StreetDataset(root_dir=CONFIG["root_dir"], split="test", resolution=CONFIG["img_size"]) # Note: 'test' here
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(f"🔥 Starting Training! Results will be saved to: {CONFIG['output_dir']}")

    for epoch in range(CONFIG["n_epochs"]):
        generator.train()
        discriminator.train()
        
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['n_epochs']}")
        
        for i, batch in enumerate(tqdm_bar):
            real_A = batch["A"].to(device) # Sat (Polar)
            real_B = batch["B"].to(device) # Street

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            
            patch = (1, real_A.size(2) // 16, real_A.size(3) // 16)
            valid = torch.ones((real_A.size(0), *patch), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), *patch), requires_grad=False).to(device)

            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            loss_G = loss_GAN + CONFIG["lambda_pixel"] * loss_pixel
            
            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            
            loss_D.backward()
            optimizer_D.step()

            tqdm_bar.set_postfix(G=loss_G.item(), D=loss_D.item())

        # --- Validation (Every 5 epochs) ---
        if (epoch + 1) % 5 == 0 or epoch == 0:
            generator.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                val_real_A = val_batch["A"].to(device)
                val_real_B = val_batch["B"].to(device)
                val_fake_B = generator(val_real_A)
                
                img_sample = torch.cat((val_real_A, val_fake_B, val_real_B), -1)
                save_path = os.path.join(CONFIG["output_dir"], f"epoch_{epoch+1}.png")
                save_image(img_sample, save_path, nrow=2, normalize=True)
                print(f"🖼️ Validation image saved: {save_path}")
            
            torch.save(generator.state_dict(), os.path.join(CONFIG["output_dir"], f"generator_epoch_{epoch+1}.pth"))
