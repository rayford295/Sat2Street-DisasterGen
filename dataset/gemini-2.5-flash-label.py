import os
import base64
import time
import json
import pandas as pd
import google.generativeai as genai
from tqdm.auto import tqdm
from PIL import Image

# ================= Configuration Area =================
# ⚠️ Please replace with your own API KEY
API_KEY = "your api" 
IMAGE_FOLDER = "./images"   # Path to your image folder
OUTPUT_CSV = "./captions.csv"
MODEL_NAME = "gemini-2.5-flash" 

# Configure API
genai.configure(api_key=API_KEY)
# ======================================================

def encode_image(image_path):
    """Reads image and converts to the format required by Gemini API."""
    with open(image_path, "rb") as image_file:
        return {"mime_type": "image/jpeg", "data": image_file.read()}

def get_gemini_annotation(model, image_path):
    """
    Sends image to Gemini and retrieves structured annotations.
    """
    img_data = encode_image(image_path)
    
    # 🔥 Core Prompt: Instructs Gemini on how to analyze the image 🔥
    # We request the result in JSON format for easier downstream processing
    prompt = """
    You are an expert in remote sensing and disaster assessment. 
    Analyze this satellite image (top-down view) of a disaster-affected area.
    
    Your task is to generate information to guide a text-to-image model (like Stable Diffusion) 
    to reconstruct the corresponding street-level view.

    Return the result strictly in JSON format with the following keys:
    1. "severity": Choose one from ["Mild", "Moderate", "Severe"].
       - Mild: Intact roofs, clear roads, minor debris.
       - Moderate: Some damaged roofs (missing shingles), visible debris on roads, some fallen trees.
       - Severe: Structural collapse, destroyed roofs (blue tarps or holes), flooded roads, massive debris.
    2. "description": A highly detailed visual prompt for a street-view image. 
       Focus on: texture of the ground (muddy, flooded, paved), state of buildings (roof condition, broken windows), 
       vegetation (snapped trees, green), and weather/lighting implied by the scene. 
       Start with "A street level photo of..."
    
    Example Output format:
    {"severity": "Severe", "description": "A street level photo of a hurricane damaged neighborhood, destroyed houses with missing roofs, piles of wooden debris on the wet asphalt road, snapped palm trees, overcast sky."}
    """

    try:
        response = model.generate_content([prompt, img_data])
        # Attempt to clean the returned text to ensure it is pure JSON
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        return None

def main():
    # 1. Find all _sat images
    all_files = os.listdir(IMAGE_FOLDER)
    sat_files = [f for f in all_files if f.endswith("_sat.jpg") or f.endswith("_sat.png")]
    
    print(f"🚀 Found {len(sat_files)} satellite images. Starting annotation with {MODEL_NAME}...")

    # Initialize model
    model = genai.GenerativeModel(MODEL_NAME)
    
    results = []
    
    # 2. Iterate and process (with progress bar)
    for filename in tqdm(sat_files):
        file_path = os.path.join(IMAGE_FOLDER, filename)
        
        # Call API
        annotation = get_gemini_annotation(model, file_path)
        
        if annotation:
            # Find the corresponding street-view filename (assuming naming rule differs only by suffix)
            # This step helps with future merging into pairs.csv
            svi_filename = filename.replace("_sat", "_svi")
            
            results.append({
                "sat_path": filename,
                "svi_path": svi_filename, # Reserved for alignment
                "severity": annotation.get("severity", "Unknown"),
                "description": annotation.get("description", "")
            })
            
        # ⚠️ Free API has rate limits, appropriate sleep is recommended
        # If you have a Pay-as-you-go account, you can remove or reduce this sleep
        time.sleep(1.5) 

    # 3. Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Done! Annotations saved to {OUTPUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
