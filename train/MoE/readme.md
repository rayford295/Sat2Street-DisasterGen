# 🌍 Disaster-MoE: Satellite-to-Street View Synthesis
> **Decoupling Semantic Texture from Geometric Structure via Consistency-Filtered Mixture of Experts**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗%20HuggingFace-Diffusers-yellow)](https://huggingface.co/docs/diffusers/index)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

This repository hosts the official implementation of the **Consistency-Filtered Mixture of Experts (MoE)** pipeline. 

Our framework addresses the critical challenge in disaster response: **generating ground-level street views from post-disaster satellite imagery.** Unlike traditional methods (Pix2Pix, ControlNet) that suffer from "averaging effects" due to viewpoint ambiguity, our approach employs a **Router-Expert architecture**:

1.  **Router (The Brain):** Diagnoses damage severity based purely on visible satellite textures.
2.  **Experts (The Hands):** Three specialized ControlNets (Mild, Moderate, Severe) generate high-fidelity textures.
3.  **Consistency Filtering:** A novel data engineering strategy that removes hallucinated labels by aligning Ground Truth with VLM (Gemini) predictions.

---

## 🏗️ Architecture Pipeline

```mermaid
graph TD
    A[Satellite Image] --> B{Consistency Filter}
    B -->|"GT != Gemini"| C[Discard / Noise]
    B -->|"GT == Gemini"| D[Router Training Data]
    
    D --> E[Train Router - ResNet18]
    
    subgraph Inference ["Inference / Generation"]
    F[New Satellite Input] --> E
    E -->|Weights| G[Mixture of Experts]
    G --> H[Expert: Mild]
    G --> I[Expert: Moderate]
    G --> J[Expert: Severe]
    H --> K[Generated Street View]
    I --> K
    J --> K
    end
```end
