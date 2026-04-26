import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

CLASSES = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain",
    "sky","person","rider","car","truck","bus","train",
    "motorcycle","bicycle"
]

PALETTE = [
    [128,64,128],[244,35,232],[70,70,70],[102,102,156],
    [190,153,153],[153,153,153],[250,170,30],[220,220,0],
    [107,142,35],[152,251,152],[70,130,180],[220,20,60],
    [255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],
    [0,0,230],[119,11,32]
]

class Segmentor:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            use_fast=False)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(device)
        self.model.eval()

    def predict(self, img_pil):
        inputs = self.processor(images=img_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        seg_map = F.interpolate(
            outputs.logits,
            size=(img_pil.size[1], img_pil.size[0]),
            mode="bilinear", align_corners=False
        ).argmax(dim=1).squeeze().cpu().numpy()
        return seg_map