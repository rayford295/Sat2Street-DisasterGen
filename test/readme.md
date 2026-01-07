# 🧪 Satellite-to-Street View Inference & Evaluation Pipeline

This directory contains the complete testing suite for the **Disaster-MoE** project. It includes scripts to generate synthetic street view imagery using four distinct methodologies, strictly following a standard evaluation protocol on the test dataset (300 samples across 3 severity classes).

## 📂 File Overview

| File Name | Description |
| :--- | :--- |
| **`test_dataset_preparation_and_result_saving.py`** | **[Core Utility]** Handles data loading, path mapping (Local vs. Cloud paths), polar coordinate transformations, and the standardized saving logic (Resizing to original dimensions). |
| **`pix2pix_inference_pipeline.py`** | **[Baseline 1]** Inference script for the GAN-based Pix2Pix model. |
| **`controlnet_sd15_inference_pipeline.py`** | **[Baseline 2]** Inference for Standard Stable Diffusion 1.5 + ControlNet (using generic prompts). |
| **`controlnet_sd15_gemini_prompt_inference_pipeline.py`** | **[Baseline 3]** Inference using SD 1.5 + ControlNet enhanced with **Gemini VLM semantic prompts**. |
| **`moe_controlnet_router_inference_pipeline.py`** | **[Ours]** The **Mixture of Experts (MoE)** inference pipeline, utilizing the Router to dynamically weight experts (Mild/Moderate/Severe). |
| **`image_synthesis_evaluation_metrics.py`** | **[Evaluation]** Calculates quantitative metrics (SSIM, PSNR, LPIPS, FID) to compare all methods. |

---

## ⚙️ Experimental Setup

### 1. Data Configuration
The pipeline is configured to process the **Test Split** consisting of **300 image pairs** (100 Mild, 100 Moderate, 100 Severe).

* **Data Root:** `D:\yifan_2025\data\images` (Contains both Satellite and Ground Truth Street Views)
* **Test List:** `data/test_split.csv`
* **Gemini Captions:** `data/captions.csv` (For Method 3 & 4)

### 2. Post-Processing & Output Structure
To ensure fair qualitative comparison, all generated images undergo a **post-processing step**:
1.  **Generation:** Model generates a 512x512 (or 256x256) image.
2.  **Restoration:** The generated image is resized to match the **exact dimensions** of the original Ground Truth street view.
3.  **Organization:** Results are saved hierarchically by Method and Damage Category.

**Output Directory Structure:**
```text
D:\yifan_2025\evaluation_results\
├── output_pix2pix\
│   ├── Mild\      (Contains: 001_real.png, 001_gen.png...)
│   ├── Moderate\
│   └── Severe\
├── output_controlnet_sat2street\
├── output_controlnet_gemini_prompt\
└── moe_processed_data\ (Ours)
