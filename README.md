# StreetSynth
**Human-in-the-Loop GAN for Inclusive Street Design**

Author: Akhil Chivukula | PRN: 23070126009 | AIML A1

## Pipeline
Input Photo → SegFormer-B5 → MiDaS → PlacementEngine → LaMa → AccessNet → Output

## Setup
```bash
pip install -r requirements.txt
python pipeline.py --image test_images/sample.jpg --intervention crosswalk
```

## Novel Contributions (AccessNet)
- 5-channel input: RGB + segmentation + depth
- Multi-scale PatchGAN discriminator (70×70 + 140×140)
- Depth consistency loss