# 🌍 Satellite-to-Street: Synthesizing Post-Disaster Views from Satellite Imagery via Generative Vision Models

> **A Benchmark Study of Generative Models for Post-Disaster Satellite-to-Street View Synthesis**  
> *(Including Pix2Pix, Diffusion + ControlNet, VLM-Guided Generation, and Mixture-of-Experts)*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗%20HuggingFace-Diffusers-yellow)](https://huggingface.co/docs/diffusers/index)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---
## 📢 News

🎉 **Paper accepted at IGARSS 2026!**  
We are excited to announce that our work on **satellite-to-street view synthesis for post-disaster assessment** has been accepted at the  
**IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2026)**.

## 📖 Overview

This repository provides a **comprehensive benchmark and analysis framework** for **post-disaster satellite-to-street view image synthesis**.

Ground-level street-view imagery is crucial for disaster damage assessment, yet often unavailable immediately after disasters due to access constraints. This project investigates whether **realistic and semantically consistent street views can be synthesized from post-disaster satellite imagery**, and how different generative paradigms perform under this challenging cross-view setting.

Unlike repositories that focus on a single model, **this project systematically evaluates multiple generation strategies**, highlighting their respective strengths and limitations in disaster scenarios.

---

## 🔍 Evaluated Methods

We benchmark a set of representative **satellite-to-street-view synthesis approaches** under post-disaster scenarios.  
These methods span conditional GANs, diffusion-based models, semantic-guided generation, and expert-based routing strategies.

### 1️⃣ Pix2Pix (Conditional GAN)

- A classical conditional GAN for image-to-image translation  
- Effective at preserving coarse spatial structure  
- Prone to texture blurring and mode collapse in complex disaster scenes  

### 2️⃣ Stable Diffusion 1.5 + ControlNet

- Diffusion-based generation guided by satellite-derived structural cues  
- ControlNet enforces geometric consistency between overhead and ground-level views  
- Produces highly realistic images, but may underestimate fine-grained damage patterns  

### 3️⃣ ControlNet + Vision–Language Semantic Guidance (Gemini)

- Incorporates high-level disaster semantics via a vision–language model  
- Improves damage awareness and semantic expressiveness  
- May introduce semantic–geometry tension when inferred semantics conflict with structural constraints  

### 4️⃣ Disaster-MoE (Mixture of Experts)

- A severity-aware generative variant evaluated in this repository  
- Decouples geometric structure from semantic texture  
- Adapts generation behavior across mild, moderate, and severe damage conditions  

> ⚠️ **Important Note**  
> Disaster-MoE is included as **one of the evaluated methods**, rather than the sole contribution of this repository.  
> The primary goal is **comparative benchmarking** and understanding trade-offs among different cross-view disaster synthesis strategies.

---

## 🧭 Overall Methodology

The following figure presents the **unified evaluation framework** adopted in this repository for satellite-to-street-view synthesis under post-disaster scenarios.  
All evaluated methods share identical satellite inputs, data splits, and evaluation protocols, ensuring a fair and controlled comparison.

<p align="center">
  <img src="Figure/method.drawio.png" alt="Overall methodology for satellite-to-street-view disaster synthesis" width="90%">
</p>


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
**BibTeX**
</td>
<td>

```bibtex
@article{yang2026satellite,
  title   = {Satellite-to-Street: Synthesizing Post-Disaster Views from Satellite Imagery via Generative Vision Models},
  author  = {Yang, Yifan and Zou, Lei and Jepson, Wendy},
  journal = {arXiv preprint arXiv:2603.20697},
  year    = {2026}
}
```


**Plain Text**

</td>
<td>

Yang, Y., Zou, L., & Jepson, W. (2026). Satellite-to-Street: Synthesizing Post-Disaster Views from Satellite Imagery via Generative Vision Models. *arXiv preprint arXiv:2603.20697*.

</td>
</tr>
<tr>
<td>


## 🙏 Acknowledgement

This project was supported by the Texas A&M University Environment and Sustainability Initiative (ESI) through the Environment and Sustainability Graduate Fellow Award.


