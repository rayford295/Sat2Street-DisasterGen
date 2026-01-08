# 🌍 Sat2Street-DisasterGen: Satellite-to-Street View Synthesis for Post-Disaster Assessment

> **A Benchmark Study of Generative Models for Post-Disaster Satellite-to-Street View Synthesis**  
> *(Including Pix2Pix, Diffusion + ControlNet, VLM-Guided Generation, and Mixture-of-Experts)*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗%20HuggingFace-Diffusers-yellow)](https://huggingface.co/docs/diffusers/index)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

This repository provides a **comprehensive benchmark and analysis framework** for **post-disaster satellite-to-street view image synthesis**.

Ground-level street-view imagery is crucial for disaster damage assessment, yet often unavailable immediately after disasters due to access constraints. This project investigates whether **realistic and semantically consistent street views can be synthesized from post-disaster satellite imagery**, and how different generative paradigms perform under this challenging cross-view setting.

Unlike repositories that focus on a single model, **this project systematically evaluates multiple generation strategies**, highlighting their respective strengths and limitations in disaster scenarios.

---

## 🔍 Evaluated Methods

We benchmark the following representative approaches:

### 1️⃣ Pix2Pix (Conditional GAN)
- Strong at preserving coarse spatial structure  
- Prone to texture blurring and mode collapse in complex disaster scenes  

### 2️⃣ Stable Diffusion 1.5 + ControlNet
- Geometry-guided diffusion using satellite-derived structural cues  
- Produces the most visually realistic images overall  

### 3️⃣ ControlNet + VLM Semantic Guidance (Gemini)
- Injects high-level disaster semantics via vision–language models  
- Improves damage awareness but may introduce semantic–geometry tension  

### 4️⃣ Disaster-MoE (Mixture of Experts) **[One Method in This Repo]**
- Decouples geometric structure from semantic texture  
- Routes samples to severity-specific experts (Mild / Moderate / Severe)  
- Improves realism and semantic diversity under severe damage conditions  

> ⚠️ **Important:**  
> **Disaster-MoE is one of the evaluated methods**, not the sole contribution of this repository.  
> The main goal is **comparative analysis and understanding trade-offs** in cross-view disaster synthesis.

---

## 🧠 Disaster-MoE Architecture (Optional Method)

```mermaid
graph TD
    A[Satellite Image] --> B{Consistency Filter}
    B -->|"GT ≠ Gemini"| C[Discard / Noise]
    B -->|"GT = Gemini"| D[Clean Training Set]
    D --> E[Router Training - ResNet18]

    subgraph Inference ["Inference / Generation"]
        F[Satellite Input] --> E
        E -->|Routing Weights| G[Mixture of Experts]
        G --> H[Expert: Mild]
        G --> I[Expert: Moderate]
        G --> J[Expert: Severe]
        H --> K[Generated Street View]
        I --> K
        J --> K
    end


````

---

## 🖼️ Qualitative Results

### Satellite-to-Street View Synthesis Across Damage Severities

The following figure presents a qualitative comparison of synthesized street-view images under **Mild**, **Moderate**, and **Severe** disaster conditions across different generative models.

![Qualitative Comparison](https://raw.githubusercontent.com/rayford295/Sat2Street-DisasterGen/main/Figure/compare%20result.PNG)

**Key observations:**

* **GAN-based models (Pix2Pix)** preserve coarse structural layouts but suffer from blurred textures and limited realism.
* **Diffusion-based models (SD1.5 + ControlNet)** generate more photorealistic textures, but may hallucinate intact structures in heavily damaged areas.
* **Semantic guidance (Gemini) and Mixture-of-Experts (MoE)** improve damage-aware synthesis, especially in *moderate* and *severe* disaster scenarios.

---

## 📊 Semantic Consistency Evaluation

To assess whether synthesized images preserve **disaster severity semantics**, we evaluate generated street views using a **ResNet-18 classifier** trained on real post-disaster street-view images.

![Confusion Matrices](https://raw.githubusercontent.com/rayford295/Sat2Street-DisasterGen/main/Figure/confusion_matrices_comparison.png)

**Findings:**

* **SD1.5 + ControlNet** achieves the highest semantic consistency (**F1 ≈ 0.71**), closely approaching real street-view performance.
* **Pix2Pix** exhibits strong mode collapse, predicting most samples as *Mild* damage.
* **Gemini-guided and MoE models** show improved visual realism but slightly reduced class separability, indicating a trade-off between perceptual richness and semantic consistency.


## 📌 Citation

If you use this code, data, or figures in academic work, please **cite the corresponding paper** or **contact the author** in advance.

This repository and its outputs are intended **for research and academic use only**.

A formal BibTeX citation will be released upon paper acceptance.

