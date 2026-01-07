import os
import json
import time
import random
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import google.generativeai as genai

# ================= 🔧 Global Configuration =================
API_KEY = "API"  # 🔥 Insert your key here
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Path configuration
EVAL_ROOT = r"D:\yifan_2025\evaluation_results"
OUTPUT_DIR = os.path.join(EVAL_ROOT, "llm_reports_full")  # Use a new folder name to avoid overwriting prior outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Method lists
OUR_METHOD = "MoE_Ours"
BASELINES = ["Pix2Pix", "SD1.5_ControlNet", "SD1.5_ControlNet_Gemini"]
ALL_METHODS = BASELINES + [OUR_METHOD]

# 🔥 Set SAMPLE_NUM = None to evaluate ALL images (e.g., all 100 per category)
SAMPLE_NUM = None

# API cooldown (seconds). For full runs, 1.5–2.0 is recommended to reduce rate-limit risk.
SLEEP_TIME = 1.5

# ================= 🧠 Prompt Library =================

# 1) Absolute scoring prompt
PROMPT_ABSOLUTE = """
You are an expert Disaster Assessment Specialist.
You are viewing two images: 
- LEFT: Ground Truth (Real)
- RIGHT: AI Generated

Evaluate the RIGHT image (Generated) based on the LEFT image (GT) on a scale of 1-5.

Dimensions:
1. Structural Consistency (1-5): Does it keep the same road/building layout?
2. Damage Severity Accuracy (1-5): Does it show the EXACT same level of damage? (Crucial: If GT is ruined, Gen must be ruined).
3. Texture Realism (1-5): How photorealistic is it?

Return JSON:
{
  "structural": <int>,
  "damage": <int>,
  "realism": <int>
}
"""

# 2) Pairwise arena prompt
PROMPT_BATTLE = """
You are a Senior Rescue Commander.
You see three images:
1. Reference (Left): Ground Truth.
2. Image A (Middle): Method A.
3. Image B (Right): Method B.

Task: Compare Image A and B against Reference. Which one better captures the **Disaster Damage Severity** and **Structural Details**?
If Reference shows rubble, the winner MUST show rubble.

Return JSON:
{
  "winner": "A" or "B" or "Tie",
  "reason": "Brief reason focusing on damage accuracy"
}
"""

# ================= 🛠️ Image Utilities =================

def resize_h(img, target_h=512):
    return img.resize((int(img.width * target_h / img.height), target_h))

def combine_pair(path1, path2):
    """Concatenate two images (Left | Right)."""
    try:
        i1 = resize_h(Image.open(path1).convert("RGB"))
        i2 = resize_h(Image.open(path2).convert("RGB"))
        combo = Image.new("RGB", (i1.width + i2.width, 512))
        combo.paste(i1, (0,0)); combo.paste(i2, (i1.width,0))
        return combo
    except Exception as e:
        print(f"⚠️ Image read error: {e}")
        return None

def combine_triplet(ref_path, path_a, path_b):
    """Concatenate three images (Ref | A | B)."""
    try:
        ref = resize_h(Image.open(ref_path).convert("RGB"))
        a = resize_h(Image.open(path_a).convert("RGB"))
        b = resize_h(Image.open(path_b).convert("RGB"))
        combo = Image.new("RGB", (ref.width + a.width + b.width, 512))
        combo.paste(ref, (0,0)); combo.paste(a, (ref.width,0)); combo.paste(b, (ref.width+a.width,0))
        return combo
    except Exception as e:
        print(f"⚠️ Image read error: {e}")
        return None

def call_gemini(prompt, image):
    if image is None: return None
    try:
        # Add retries to handle occasional network instability
        for _ in range(3):
            try:
                response = model.generate_content([prompt, image])
                text = response.text.replace("```json", "").replace("```", "").strip()
                return json.loads(text)
            except Exception:
                time.sleep(2)  # wait 2 seconds before retry
        return None
    except:
        return None

# ================= 📊 Stage 1: Absolute Scoring =================
def run_absolute_scoring():
    print("\n📊 [Stage 1] Running absolute scoring (full evaluation)...")
    results = []
    
    for method in ALL_METHODS:
        print(f"   Evaluating: {method}")
        method_dir = os.path.join(EVAL_ROOT, method)
        
        # Iterate over all categories
        for cat in ["Mild", "Moderate", "Severe"]:
            cat_dir = os.path.join(method_dir, cat)
            if not os.path.exists(cat_dir): continue
            
            # Collect all files
            files = [f for f in os.listdir(cat_dir) if f.endswith("_real.png")]
            
            # 🔥 SAMPLE_NUM is None -> process all files
            target_files = files[:SAMPLE_NUM]
            
            for f in tqdm(target_files, desc=f"    {cat}", leave=False):
                real_p = os.path.join(cat_dir, f)
                gen_p = os.path.join(cat_dir, f.replace("_real.png", "_gen.png"))
                
                if not os.path.exists(gen_p): continue
                
                img = combine_pair(real_p, gen_p)
                res = call_gemini(PROMPT_ABSOLUTE, img)
                
                if res:
                    results.append({
                        "Method": method,
                        "Category": cat,
                        "Filename": f,
                        "Struct": res.get("structural", 3),
                        "Damage": res.get("damage", 3),
                        "Realism": res.get("realism", 3)
                    })
                time.sleep(SLEEP_TIME)  # rate-limit protection
                
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "absolute_scores_full.csv"), index=False)
    return df

# ================= ⚔️ Stage 2: Pairwise Arena Battle =================
def run_pairwise_battle():
    print("\n⚔️ [Stage 2] Running pairwise arena battles (full evaluation)...")
    results = []
    
    our_dir = os.path.join(EVAL_ROOT, OUR_METHOD)
    
    for baseline in BASELINES:
        print(f"   🥊 Round: MoE_Ours vs {baseline}")
        base_dir = os.path.join(EVAL_ROOT, baseline)
        
        for cat in ["Mild", "Moderate", "Severe"]:
            cat_ours = os.path.join(our_dir, cat)
            cat_base = os.path.join(base_dir, cat)
            if not os.path.exists(cat_ours) or not os.path.exists(cat_base): continue
            
            files = [f for f in os.listdir(cat_ours) if f.endswith("_real.png")]
            target_files = files[:SAMPLE_NUM]  # full set
            
            for f in tqdm(target_files, desc=f"    {cat}", leave=False):
                ref_p = os.path.join(cat_ours, f)
                our_gen = os.path.join(cat_ours, f.replace("_real.png", "_gen.png"))
                base_gen = os.path.join(cat_base, f.replace("_real.png", "_gen.png"))
                
                if not os.path.exists(our_gen) or not os.path.exists(base_gen): continue
                
                # Randomly swap A/B positions to reduce positional bias
                swap = random.random() > 0.5
                img_a, img_b = (base_gen, our_gen) if swap else (our_gen, base_gen)
                mapping = {"A": baseline, "B": OUR_METHOD} if swap else {"A": OUR_METHOD, "B": baseline}
                
                img = combine_triplet(ref_p, img_a, img_b)
                res = call_gemini(PROMPT_BATTLE, img)
                
                if res:
                    raw_winner = res.get("winner", "Tie")
                    real_winner = mapping.get(raw_winner, "Tie")
                    
                    results.append({
                        "Opponent": baseline,
                        "Category": cat,
                        "Filename": f,
                        "Winner": real_winner,
                        "Reason": res.get("reason", "")
                    })
                time.sleep(SLEEP_TIME)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "pairwise_win_rates_full.csv"), index=False)
    return df

# ================= 📋 Auto-generate summary report =================
def generate_final_report(df_abs, df_battle):
    print("\n" + "="*50)
    print("🏆 DISASTER-MOE FINAL EVALUATION REPORT (FULL RUN)")
    print("="*50)
    
    if df_abs is not None and not df_abs.empty:
        print("\n1️⃣ Absolute Scores (1–5) — Averages:")
        summary = df_abs.groupby("Method")[["Struct", "Damage", "Realism"]].mean()
        print(summary.to_string(float_format="{:.2f}".format))
        
    if df_battle is not None and not df_battle.empty:
        print("\n2️⃣ Arena Win Rates (MoE Win Rate):")
        for base in BASELINES:
            subset = df_battle[df_battle["Opponent"] == base]
            if len(subset) == 0: continue
            wins = len(subset[subset["Winner"] == OUR_METHOD])
            ties = len(subset[subset["Winner"] == "Tie"])
            total = len(subset)
            
            win_rate = (wins / total) * 100
            tie_rate = (ties / total) * 100
            
            print(f"   vs {base:25s}: MoE Wins {win_rate:5.1f}% (Ties: {tie_rate:.1f}%) | Count: {total}")
            
    print("\n✅ All detailed outputs have been saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    # 1) Absolute scoring
    df_abs = run_absolute_scoring()
    
    # 2) Pairwise battles
    df_battle = run_pairwise_battle()
    
    # 3) Summary
    generate_final_report(df_abs, df_battle)
