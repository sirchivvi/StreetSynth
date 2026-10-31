---
title: StreetSynth
emoji: 🏙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# StreetSynth: A Human-in-the-Loop GAN Framework for Inclusive Urban Street Design Visualization

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/SirChivvi/StreetSynth)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sirchivvi/StreetSynth/blob/main/notebooks/StreetSynth_Colab_Inference.ipynb)
[![Deployment: GitHub Pages](https://img.shields.io/badge/Deployment-GitHub%20Pages-222222.svg?logo=github)](https://sirchivvi.github.io/StreetSynth/)

**Author:** Akhil Chivukula &nbsp;|&nbsp; **PRN:** 23070126009 &nbsp;|&nbsp; **Division:** AIML A1  
*Artificial Intelligence & Machine Learning Specialization*

---

## Overview

StreetSynth is an AI-powered urban planning visualization system that generates photorealistic accessibility interventions from a single street photograph. The project combines semantic segmentation, depth estimation, rule-based reasoning, image inpainting, and generative adversarial networks to help planners, researchers, and community stakeholders visualize accessibility improvements before implementation.

The system supports three common accessibility interventions:

- **Pedestrian Crosswalks** (connectivity across roadways)
- **Public Benches** (pedestrian comfort and resting stops)
- **Curb Ramps** (wheelchair mobility transitions)

Unlike traditional image generation systems, StreetSynth incorporates an explainable rule-based placement engine that ensures interventions are placed in spatially valid and accessibility-compliant locations before synthesis.

---

## Problem Statement

Urban accessibility planning often depends on expensive architectural visualizations, CAD software, and manual design workflows. These approaches are difficult for non-technical stakeholders to understand and require significant expertise.

StreetSynth addresses this challenge by automatically answering a simple question:

> "What would this street look like if accessibility improvements were added?"

Given a street image, the system identifies suitable locations for accessibility interventions and generates a realistic visualization of the modified environment.

---

## Key Features

### Human-in-the-Loop Design
Users select an intervention type and review AI-generated placement suggestions rather than allowing fully autonomous generation.

### Explainable Placement Engine
Every placement decision is backed by interpretable geometric and semantic rules.

### Multi-Modal Scene Understanding
The system combines RGB imagery, semantic segmentation, and monocular depth estimation to understand both scene structure and geometry.

### Photorealistic Generation
A custom GAN architecture synthesizes realistic interventions that blend naturally into existing urban scenes.

### $0 Budget Free Deployment
- **Research Showcase:** Hosted globally 24/7 on **GitHub Pages** with zero server costs.
- **Hardware-Accelerated Inference:** Available on-demand via **Google Colab** (Free T4 GPU) and **Hugging Face Spaces**.

---

# System Architecture

```text
Street Image (RGB 1024×512)
      │
      ▼
Semantic Segmentation
(DeepLabV3+ / SegFormer-B5)
      │
      ▼
Depth Estimation
(MiDaS / DPT-Large)
      │
      ▼
Accessibility Placement Engine
(Rule-Based Geometric Reasoning)
      │
      ▼
LaMa Inpainting
(Clean Canvas Preparation)
      │
      ▼
AccessNet GAN
(5-Channel Conditional U-Net)
      │
      ▼
Final Accessibility Visualization
```

---

# Methodology

## 1. Semantic Segmentation

StreetSynth performs pixel-level scene understanding using a pretrained SegFormer-B5 / DeepLabV3+ model on Cityscapes.

Important urban classes include:
- Road
- Sidewalk
- Building
- Vehicle
- Pole
- Person
- Vegetation

The segmentation map provides semantic awareness required for accessibility-aware placement decisions.

---

## 2. Depth Estimation

The system uses MiDaS / DPT-Large to generate relative depth maps.

Depth information enables:
- Surface flatness estimation
- Distance reasoning
- Geometric consistency
- Perspective-aware intervention placement

This stage allows the system to understand not only what objects exist in the scene but also where they exist spatially.

---

## 3. Accessibility Placement Engine

The placement engine is the primary novel contribution of the project.

Instead of randomly inserting interventions, StreetSynth uses accessibility-aware geometric constraints (`cv/placement.py`):

### Crosswalk Rules
- Must connect road and sidewalk boundaries
- Must satisfy minimum span requirements (`span > 0.15 * W`)
- Must be located on relatively flat ground (`σ_depth < 0.05`)

### Bench Rules
- Must be located on an unobstructed sidewalk
- Must avoid pedestrians, vehicles, and poles
- Must not occupy more than 20% of sidewalk width to guarantee a 1.8m ADA clear corridor

### Curb Ramp Rules
- Must be positioned on road-sidewalk boundaries
- Must align with pedestrian crossing regions (80th-percentile road-sidewalk interface)
- Standardized tactile ramp aspect ratio

The engine outputs:
- Placement Mask
- Validity Score
- Human-readable reasoning

---

## 4. LaMa Inpainting

Before generation, the selected placement region is erased using LaMa to produce a clean background canvas. For low-resource CPU hosting, lightweight mode allows direct synthesis.

---

## 5. AccessNet Architecture

AccessNet is a conditional generative adversarial network that synthesizes realistic accessibility interventions:

- **5-Channel Input:** RGB (3) + Normalized Semantics (1) + Relative Depth (1)
- **U-Net Generator:** Skip connections with bottleneck dropout for structural preservation
- **Multi-Scale PatchGAN Discriminator:** Evaluates realism at 1x and 0.5x spatial scales
- **Multi-Term Joint Loss:** Adversarial loss + VGG-16 Perceptual L1 loss + Edge-Aware Depth Consistency loss

---

# Results

## Quantitative Performance

| Metric | Value | Baseline Comparison |
|----------|:----------:|---|
| **LPIPS Perceptual Distance** | **0.1704** | 0.2780 (Pix2Pix) |
| **Mean IoU** | **0.667** | 0.640 (Baseline) |
| **Road IoU** | **0.768** | 0.720 (Baseline) |
| **Sidewalk IoU** | **0.567** | 0.510 (Baseline) |
| **Constraint Compliance Rate** | **95.0%** | 54.0% (Unconstrained) |
| **Accessibility Score** | **3 / 3** | 1.2 / 3 (Unguided) |

---

## Training Convergence

| Epoch | Generator Loss | L1 Loss | Perceptual Loss |
|:---------:|:---------:|:---------:|:---------:|
| 1 | 14.39 | 0.080 | 0.350 |
| 5 | 4.45 | 0.019 | 0.130 |
| 10 | 3.86 | 0.015 | 0.100 |
| 15 | 3.47 | 0.013 | 0.087 |
| 20 | 3.29 | 0.011 | 0.083 |

*Training remained stable throughout all epochs without mode collapse.*

---

## Inference Performance

### Latency
- **GPU (CUDA):** 8–10 seconds per image
- **CPU (8-Core):** 30–60 seconds per image

### Memory Footprint
- **GPU:** < 4 GB VRAM
- **CPU:** < 3 GB RAM

---

## Ablation Study

| Configuration | Outcome |
|--------------|----------|
| **L1 Only** | Blurry outputs |
| **L1 + Perceptual** | Improved textures |
| **L1 + Depth** | Better geometric consistency |
| **Full AccessNet** | Best overall realism |

**Key findings:**
- Perceptual loss improves texture fidelity.
- Depth consistency loss improves spatial realism.
- Combining all losses produces the highest-quality outputs.

---

# 🚀 Deployment Guide ($0 Budget)

### 1. GitHub Pages Showcase Deployment (Automated)
1. In your GitHub repository, go to **Settings > Pages**.
2. Under **Build and deployment > Source**, select **GitHub Actions**.
3. Pushing to `main` triggers `.github/workflows/deploy.yml` and publishes to:
   ```
   https://<your-username>.github.io/StreetSynth/
   ```

### 2. Free GPU Inference on Google Colab
1. Open [`notebooks/StreetSynth_Colab_Inference.ipynb`](notebooks/StreetSynth_Colab_Inference.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU** (Free tier).
3. Run all cells to execute synthesis in 8–10 seconds.

### 3. Local Development Setup
```bash
# 1. Start Local FastAPI Backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 2. Start Frontend Dev Server (Vite)
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

---

## 📁 Repository Structure

```
StreetSynth/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions workflow for GitHub Pages ($0 hosting)
├── api/
│   └── main.py                 # FastAPI backend for local/optional inference
├── checkpoints/
│   └── accessnet_ep020.pt      # AccessNet PyTorch generator checkpoint (568MB)
├── cv/
│   ├── depth.py                # DPT-Large / MiDaS monocular depth estimation
│   ├── indicators.py           # Accessibility rule-based scoring (0-3)
│   ├── placement.py            # Geometric feasibility & boundary engine
│   └── segmentation.py         # SegFormer-B5 Cityscapes 19-class parser
├── frontend/
│   ├── public/samples/         # Bundled high-resolution Cityscapes outputs
│   ├── src/
│   │   ├── components/         # BeforeAfterSlider, MetricCard, Navbar, Footer
│   │   ├── pages/              # Home, Demo, Metrics, Methodology, About
│   │   ├── data/mockData.ts    # Verified empirical benchmarks & constraints
│   │   ├── App.tsx             # HashRouter navigation & health detection
│   │   └── main.tsx
│   ├── package.json
│   ├── tailwind.config.js      # Dark Charcoal (#101412) + Emerald (#10B981)
│   └── vite.config.ts          # Configured with relative base for GitHub Pages
├── gan/
│   ├── discriminator.py        # Multi-Scale PatchGAN
│   ├── generator.py            # AccessNet 5-channel U-Net
│   └── losses.py               # Adversarial, Perceptual L1, Depth Consistency
├── notebooks/
│   └── StreetSynth_Colab_Inference.ipynb  # Free Colab T4 GPU inference notebook
├── outputs/                    # Pre-computed sample outputs & comparisons
├── ui/
│   └── app.py                  # Gradio application for Hugging Face Spaces
├── app.py                      # Hugging Face Spaces launch entry point
├── Dockerfile                  # Containerized runtime configuration
├── pipeline.py                 # End-to-end StreetSynth inference pipeline
└── requirements.txt            # Python dependencies
```

---

## 📜 Academic Attribution & License

- **Author:** Akhil Chivukula (PRN: `23070126009`)
- **Institution:** Department of Artificial Intelligence and Machine Learning (AIML A1)
- **License:** [MIT License](LICENSE)
- **Repository:** [https://github.com/sirchivvi/StreetSynth](https://github.com/sirchivvi/StreetSynth)
- **Live Showcase:** [https://sirchivvi.github.io/StreetSynth/](https://sirchivvi.github.io/StreetSynth/)
