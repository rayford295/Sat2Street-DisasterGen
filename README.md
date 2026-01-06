# 🌍 Disaster-MoE: Satellite-to-Street View Synthesis

> **Decoupling Semantic Texture from Geometric Structure via Consistency-Filtered Mixture of Experts**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗%20HuggingFace-Diffusers-yellow)](https://huggingface.co/docs/diffusers/index)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

This repository hosts the official implementation of **Disaster-MoE**, a framework designed to generate ground-level street views from post-disaster satellite imagery.

Generating street views from satellite data is challenging due to the **"one-to-many" ambiguity** (e.g., a gray roof in satellite view could be an intact building or a collapsed ruin). Traditional methods like Pix2Pix or standard ControlNet often suffer from "averaging effects," resulting in blurry or hallucinated structures.

**Our Solution:** A **Consistency-Filtered Mixture of Experts (MoE)** architecture that:
1. **Decouples** geometry (via ControlNet) from semantic texture (via Experts).
2. **Filters** label noise using a VLM (Gemini) consistency check.
3. **Dynamically routes** generation tasks to specialized experts (Mild, Moderate, Severe).

---

## 🏗️ System Architecture

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
