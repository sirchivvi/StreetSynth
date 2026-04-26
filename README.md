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

# StreetSynth
**Human-in-the-Loop GAN for Inclusive Street Design**

Author: Akhil Chivukula | PRN: 23070126009 | AIML A1

## Pipeline
Input Photo → SegFormer-B5 → MiDaS → PlacementEngine → LaMa → AccessNet → Output

## Novel Contributions (AccessNet)
- 5-channel input: RGB + segmentation + depth
- Multi-scale PatchGAN discriminator
- Depth consistency loss

## Setup
```bash
pip install -r requirements.txt
python pipeline.py --image test_images/sample.jpg --intervention crosswalk
```