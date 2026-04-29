import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

class DepthEstimator:
    def __init__(self, device="cuda"):
        self.device = device

        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, img_np):
        inputs = self.processor(images=img_np, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth = depth.cpu().numpy()

        # normalize
        return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)