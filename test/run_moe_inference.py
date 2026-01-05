import torch
import os
import cv2
import re
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import matplotlib.pyplot as plt

# ================= 🔧 Configuration =================
BASE_DIR = r"D:\yifan_2025\data"

# 1. Expert Model Paths (These folders exist after training)
# Note: Pointing to the 'controlnet' subfolder inside the checkpoint
PATH_MILD = os.path.join(BASE_DIR, "moe_checkpoints", "expert_mild", "checkpoint-epoch-20", "controlnet")
PATH_MOD  = os.path.join(BASE_DIR, "moe_checkpoints", "expert_moderate", "checkpoint-epoch-20", "controlnet")
PATH_SEV  = os.path.join(BASE_DIR, "moe_checkpoints", "expert_severe", "checkpoint-epoch-20", "controlnet")

# 2. Router Path (Trained in Step 2)
PATH_ROUTER = os.path.join(BASE_DIR, "moe_checkpoints", "router_consistent.pth")

# 3. Gemini Captions File
CAPTIONS_CSV = os.path.join(BASE_DIR, "captions.csv")

# 4. 🔥 Input Test Image Path Here!
# (Pick an unseen image from your images folder)
TEST_IMG_PATH = os.path.join(BASE_DIR, "images", "003895_sat.png") 

# ============================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🧹 Prompt Sanitizer: Removes conflicting terms from Gemini captions
def sanitize_prompt(text):
    if not isinstance(text, str): return ""
    banned_words = [
        "severe", "mild", "moderate", "severity", 
        "grade", "level", "classification", "category", "damage assessment",
        "destroyed", "intact", "collapsed" # Remove adjectives to let ControlNet decide the look
    ]
    for word in banned_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub("", text)
    return re.sub(' +', ' ', text).strip()

def run_moe_inference():
    print("🚀 Starting MoE Joint Inference...")

    # --- 1. Loading Router ---
    print(f"🧠 Loading Router...")
    if not os.path.exists(PATH_ROUTER):
        print("❌ Router model not found! Check path.")
        return
        
    router = models.resnet18(pretrained=False)
    router.fc = torch.nn.Linear(router.fc.in_features, 3)
    router.load_state_dict(torch.load(PATH_ROUTER))
    router.to(device).eval()

    router_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 2. Loading Experts ---
    print("🤖 Loading 3 Expert Models (This may take a few seconds)...")
    try:
        cnet_mild = ControlNetModel.from_pretrained(PATH_MILD, torch_dtype=torch.float16)
        cnet_mod  = ControlNetModel.from_pretrained(PATH_MOD, torch_dtype=torch.float16)
        cnet_sev  = ControlNetModel.from_pretrained(PATH_SEV, torch_dtype=torch.float16)
    except Exception as e:
        print(f"❌ Failed to load experts: {e}")
        print("💡 Tip: Ensure all three experts are trained and paths contain 'checkpoint-epoch-20'")
        return

    # --- 3. Initializing Pipeline ---
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[cnet_mild, cnet_mod, cnet_sev], # Multi-Expert Parallel Execution
        torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload() # Save VRAM

    # --- 4. Preparing Data ---
    if not os.path.exists(TEST_IMG_PATH):
        print(f"❌ Test image not found: {TEST_IMG_PATH}")
        return

    filename = os.path.basename(TEST_IMG_PATH)
    
    # Get Gemini Description
    caption_map = {}
    if os.path.exists(CAPTIONS_CSV):
        df = pd.read_csv(CAPTIONS_CSV)
        caption_map = dict(zip(df['sat_path'].apply(lambda x: os.path.basename(str(x))), df['description']))
    
    raw_prompt = caption_map.get(filename, "street view photography, realistic")
    clean_prompt = sanitize_prompt(raw_prompt)
    final_prompt = f"{clean_prompt}, realistic, 8k, highly detailed, photorealistic"

    # Process Image (Polar Transform)
    raw_img = Image.open(TEST_IMG_PATH).convert("RGB").resize((512, 512))
    img_np = np.array(raw_img)
    center = (256, 256)
    polar_img = cv2.linearPolar(img_np, center, 256, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cond_image = Image.fromarray(polar_img)

    # --- 5. Router Prediction ---
    r_in = router_transform(cond_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = router(r_in)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0] # [Mild, Mod, Sev]
    
    print("\n" + "="*30)
    print(f"🎯 Router Diagnosis:")
    print(f"   - Mild    : {probs[0]*100:.1f}%")
    print(f"   - Moderate: {probs[1]*100:.1f}%")
    print(f"   - Severe  : {probs[2]*100:.1f}%")
    print("="*30 + "\n")
    print(f"📝 Prompt (Sanitized): {clean_prompt[:60]}...")

    # --- 6. Generation ---
    image = pipe(
        prompt=final_prompt,
        image=[cond_image, cond_image, cond_image], 
        num_inference_steps=20,
        # 🔥 Core: Weight Allocation 🔥
        controlnet_conditioning_scale=[float(probs[0]), float(probs[1]), float(probs[2])]
    ).images[0]

    # --- 7. Display Results ---
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Satellite")
    plt.imshow(raw_img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Polar Transform (Input)")
    plt.imshow(cond_image)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("MoE Generated Street View")
    plt.imshow(image)
    plt.axis("off")
    
    plt.show()

# Run Inference
run_moe_inference()
