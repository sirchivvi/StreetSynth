import torch
import numpy as np

class DepthEstimator:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = torch.hub.load(
            "intel-isl/MiDaS", "DPT_Large", trust_repo=True)
        self.model.eval().to(device)
        self.transform = torch.hub.load(
            "intel-isl/MiDaS", "transforms",
            trust_repo=True).dpt_transform

    def predict(self, img_np):
        batch = self.transform(img_np).to(self.device)
        with torch.inference_mode():
            depth = self.model(batch)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze()
        depth = depth.cpu().numpy()
        return (depth - depth.min()) / (depth.max() - depth.min())