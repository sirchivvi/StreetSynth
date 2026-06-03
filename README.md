# StreetSynth: A Human-in-the-Loop GAN Framework for Inclusive Urban Street Design Visualization

## Overview

StreetSynth is an AI-powered urban planning visualization system that generates photorealistic accessibility interventions from a single street photograph. The project combines semantic segmentation, depth estimation, rule-based reasoning, image inpainting, and generative adversarial networks to help planners, researchers, and community stakeholders visualize accessibility improvements before implementation.

The system supports three common accessibility interventions:

- Pedestrian Crosswalks
- Public Benches
- Curb Ramps

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

### Public Deployment
Accessible through a browser using Gradio and Hugging Face Spaces.

---

# System Architecture

```text
Street Image
      │
      ▼
Semantic Segmentation
(SegFormer-B5)
      │
      ▼
Depth Estimation
(MiDaS DPT-Large)
      │
      ▼
Accessibility Placement Engine
(Rule-Based Reasoning)
      │
      ▼
LaMa Inpainting
      │
      ▼
AccessNet GAN
      │
      ▼
Final Accessibility Visualization
```

---

# Methodology

## 1. Semantic Segmentation

StreetSynth performs pixel-level scene understanding using a pretrained SegFormer-B5 model.

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

The system uses MiDaS DPT-Large to generate relative depth maps.

Depth information enables:

- Surface flatness estimation
- Distance reasoning
- Geometric consistency
- Perspective-aware intervention placement

This stage allows the system to understand not only what objects exist in the scene but also where they exist spatially.

---

## 3. Accessibility Placement Engine

The placement engine is the primary novel contribution of the project.

Instead of randomly inserting interventions, StreetSynth uses accessibility-aware geometric constraints.

### Crosswalk Rules

- Must connect two sidewalks
- Must lie on a road surface
- Must satisfy minimum width requirements
- Must be located on relatively flat ground

### Bench Rules

- Must be located on a sidewalk
- Must avoid pedestrians and vehicles
- Must preserve pedestrian movement space
- Must not occupy more than 20% of sidewalk width

### Curb Ramp Rules

- Must be positioned on road-sidewalk boundaries
- Must align with pedestrian crossing regions
- Must be placed near the user-facing portion of the scene

The engine outputs:

- Placement Mask
- Validity Score
- Human-readable reasoning

Example:

```text
Bench placement approved.

Reason:
Largest obstacle-free sidewalk region identified.
Bench occupies 12% of available walking width.
No pedestrian obstruction detected.
```

---

## 4. LaMa Inpainting

Before generation, the selected placement region is erased using LaMa.

This produces a clean background canvas that prevents visual artifacts and allows realistic intervention synthesis.

Benefits:

- Improved blending
- Reduced ghosting
- Better structural consistency

---

## 5. AccessNet GAN

AccessNet is a custom conditional GAN developed specifically for StreetSynth.

### Architecture

Based on Pix2Pix but extended with:

- Multi-modal conditioning
- Multi-scale discrimination
- Depth-aware loss functions

### Input Representation

```text
RGB Image
+
Segmentation Map
+
Depth Map
```

This allows generation to be conditioned on both semantic and geometric information.

### Generator Architecture

The generator follows a U-Net structure with:

- 7 Downsampling Blocks
- 7 Upsampling Blocks
- Skip Connections
- Instance Normalization
- Dropout Regularization

### Multi-Scale PatchGAN Discriminator

Two discriminators operate simultaneously:

#### Full Resolution

Captures:

- Fine textures
- Edge quality
- Surface detail

#### Half Resolution

Captures:

- Global structure
- Object alignment
- Perspective consistency

---

## Loss Functions

The model is optimized using four complementary objectives.

| Loss Function | Weight | Purpose |
|--------------|---------|----------|
| Adversarial Loss (LSGAN) | 1.0 | Realism |
| L1 Reconstruction Loss | 100.0 | Structure Preservation |
| VGG Perceptual Loss | 10.0 | Texture Quality |
| Depth Consistency Loss | 5.0 | Geometric Realism |

---

# Dataset

## Cityscapes Dataset

StreetSynth is trained using the Cityscapes urban scene understanding dataset.

### Dataset Statistics

- 2,975 Training Images
- 500 Validation Images
- 1,525 Test Images
- 50 European Cities

### Training Configuration

- 1,487 Images Used
- Training Resolution: 256 × 512
- Original Resolution: 1024 × 512

The dataset was selected due to its high-quality annotations and diverse urban street environments.

---

# Training Configuration

## Hardware

- NVIDIA Tesla T4 GPU
- Kaggle Free Tier

## Framework

- PyTorch

## Optimizer

```python
optimizer = Adam(
    lr=2e-4,
    betas=(0.5, 0.999)
)
```

## Training Details

- Epochs: 200
- Mixed Precision Training Enabled
- Checkpoints Saved Every 10 Epochs

---

# Results

## Quantitative Performance

| Metric | Value |
|----------|----------|
| LPIPS | 0.1704 |
| Mean IoU | 0.667 |
| Road IoU | 0.768 |
| Sidewalk IoU | 0.567 |
| Constraint Compliance Rate | 95.0% |

---

## Training Convergence

| Epoch | Generator Loss | L1 Loss | Perceptual Loss |
|---------|---------|---------|---------|
| 1 | 14.39 | 0.080 | 0.350 |
| 5 | 4.45 | 0.019 | 0.130 |
| 10 | 3.86 | 0.015 | 0.100 |
| 15 | 3.47 | 0.013 | 0.087 |
| 20 | 3.29 | 0.011 | 0.083 |

Training remained stable throughout all epochs without mode collapse.

---

## Inference Performance

### GPU

- 8–10 seconds per image

### CPU

- 30–60 seconds per image

### Memory Usage

- < 4 GB VRAM
- < 3 GB RAM

---

## Ablation Study

| Configuration | Outcome |
|--------------|----------|
| L1 Only | Blurry outputs |
| L1 + Perceptual | Improved textures |
| L1 + Depth | Better geometric consistency |
| Full AccessNet | Best overall realism |

Key findings:

- Perceptual loss improves texture fidelity.
- Depth consistency loss improves spatial realism.
- Combining all losses produces the highest-quality outputs.

---

# Technology Stack

### Deep Learning

- PyTorch
- Torchvision

### Computer Vision

- SegFormer-B5
- MiDaS DPT-Large
- OpenCV

### Generative AI

- Pix2Pix
- GANs
- LaMa

### Deployment

- Gradio
- Hugging Face Spaces

### Development Tools

- Python
- Git
- GitHub

---

# Future Work

## Additional Accessibility Features

- Tactile paving
- Accessible bus stops
- Raised crosswalks
- Adaptive traffic signals
- Wheelchair-friendly pathways

## Improved Human Interaction

- Manual placement editing
- Drag-and-drop intervention controls
- User feedback integration

## Enhanced Generation Quality

- Diffusion-based refinement
- Higher-resolution synthesis
- Larger and more diverse datasets

## Smart City Integration

- GIS connectivity
- Municipal planning workflows
- Accessibility auditing tools

---

# Conclusion

StreetSynth demonstrates how computer vision and generative AI can be combined to support inclusive urban planning. By integrating semantic understanding, depth-aware reasoning, explainable placement rules, and photorealistic image synthesis, the system provides a practical tool for visualizing accessibility interventions before real-world implementation.

The project highlights the value of combining learned models with explicit rule-based reasoning, producing outputs that are not only visually convincing but also spatially meaningful and accessibility-aware.
