# pipeline.py
import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cv.segmentation import Segmentor
from cv.depth import DepthEstimator
from cv.placement import PlacementEngine
from cv.indicators import compute_indicators, format_indicators


class StreetSynthPipeline:
    def __init__(self, checkpoint_path=None, device=None, lightweight=False):
        """
        checkpoint_path : path to AccessNet .pt checkpoint
        lightweight     : skip LaMa (for CPU/HF Spaces deployment)
        """
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lightweight = lightweight
        print(f"Device: {self.device}")

        # ── Load models ───────────────────────────────────────────────────
        print("Loading segmentor...")
        self.segmentor = Segmentor(device=self.device)

        print("Loading depth estimator...")
        self.depth_est = DepthEstimator(device=self.device)

        if not lightweight:
            print("Loading LaMa...")
            self.lama = self._load_lama()
        else:
            self.lama = None
            print("LaMa skipped (lightweight mode)")

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading AccessNet from {checkpoint_path}...")
            self.generator = self._load_generator(checkpoint_path)
        else:
            self.generator = None
            print("No AccessNet checkpoint — pipeline will run up to placement only")

        print("Pipeline ready ✓")

    # ── Model loaders ──────────────────────────────────────────────────────
    def _load_lama(self):
        import sys, yaml
        from omegaconf import OmegaConf
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lama"))

        # Patch aug.py if needed
        aug_path = os.path.join(os.path.dirname(__file__),
                                "lama/saicinpainting/training/data/aug.py")
        if os.path.exists(aug_path):
            with open(aug_path) as f:
                content = f.read()
            if "DualIAATransform" in content and "from albumentations import" in content:
                self._patch_lama_aug(aug_path)

        from saicinpainting.training.trainers import load_checkpoint

        weights_dir = os.path.join(os.path.dirname(__file__), "big-lama")
        cfg_path    = os.path.join(weights_dir, "config.yaml")
        ckpt_path   = os.path.join(weights_dir, "models", "best.ckpt")

        with open(cfg_path) as f:
            cfg = OmegaConf.create(yaml.safe_load(f))
        cfg.training_model.predict_only = True
        cfg.visualizer.kind             = "noop"

        _orig = torch.load
        def _patched(f, map_location=None, **kwargs):
            kwargs["weights_only"] = False
            return _orig(f, map_location=map_location, **kwargs)
        torch.load = _patched
        model = load_checkpoint(cfg, ckpt_path, strict=False,
                                map_location=self.device)
        torch.load = _orig
        model.freeze()
        model.to(self.device)
        return model

    def _patch_lama_aug(self, aug_path):
        patched = '''
import numpy as np
from albumentations.core.transforms_interface import DualTransform

def to_tuple(param, low=None, bias=None):
    if isinstance(param, (int, float)): return (-param, param)
    if isinstance(param, (list, tuple)): return tuple(param)
    return param

class DualIAATransform(DualTransform):
    pass

class IAAAffine2(DualIAATransform):
    def __init__(self, *args, **kwargs):
        super().__init__(always_apply=False, p=0.5)
    def apply(self, img, **params): return img
    def get_transform_init_args_names(self): return ()

class IAAPerspective2(DualIAATransform):
    def __init__(self, *args, **kwargs):
        super().__init__(always_apply=False, p=0.5)
    def apply(self, img, **params): return img
    def get_transform_init_args_names(self): return ()
'''
        with open(aug_path, "w") as f:
            f.write(patched)

    def _load_generator(self, checkpoint_path):
        from gan.generator import AccessNetGenerator
        G    = AccessNetGenerator().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        G.load_state_dict(ckpt["G_state"])
        G.eval()
        return G

    # ── Core inference ─────────────────────────────────────────────────────
    def _lama_inpaint(self, image_np, mask_np):
        from scipy.ndimage import binary_dilation
        orig_h, orig_w = image_np.shape[:2]
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        img_p  = np.pad(image_np, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
        mask_p = np.pad(mask_np,  ((0,pad_h),(0,pad_w)),       mode="reflect")

        img_t  = torch.from_numpy(img_p).float() / 255.0
        img_t  = img_t.permute(2,0,1).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_p).float().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out = self.lama({"image": img_t, "mask": mask_t})["inpainted"]

        out = out[0].permute(1,2,0).cpu().numpy()
        return np.clip(out * 255, 0, 255).astype(np.uint8)[:orig_h, :orig_w]

    def _accessnet_synthesize(self, clean_bg, seg_map, depth_norm, placement):
        """Run AccessNet generator on the clean background."""
        from scipy.ndimage import binary_dilation
        H, W = clean_bg.shape[:2]

        img_np   = clean_bg.astype(np.float32) / 255.0
        seg_norm = seg_map.astype(np.float32) / 19.0
        depth_proxy = np.linspace(1.0, 0.0, H)[:,None] * np.ones((H,W), np.float32)

        inp = np.concatenate([
            img_np,
            seg_norm[:,:,None],
            depth_proxy[:,:,None]
        ], axis=2)

        inp_t = torch.from_numpy(inp).permute(2,0,1).float().unsqueeze(0).to(self.device)
        inp_t[:,:3] = inp_t[:,:3] * 2 - 1

        with torch.inference_mode():
            out = self.generator(inp_t)

        out = out[0].permute(1,2,0).cpu().numpy()
        out = ((out + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        # Blend: only replace placement region
        mask = placement["mask"].astype(bool)
        result = clean_bg.copy()
        result[mask] = out[mask]
        return result

    # ── Main run ───────────────────────────────────────────────────────────
    def run(self, image_path, intervention_type="crosswalk", output_dir="outputs"):
        """
        Full pipeline: image → segmentation → depth → placement
                     → LaMa → AccessNet → save outputs

        Returns: dict with all intermediate and final results
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── 1. Load image ─────────────────────────────────────────────────
        print(f"\nProcessing: {image_path}")
        img_pil = Image.open(image_path).convert("RGB").resize((1024, 512))
        img_np  = np.array(img_pil)
        print(f"Image loaded: {img_np.shape}")

        # ── 2. Segmentation ───────────────────────────────────────────────
        print("Running segmentation...")
        seg_map = self.segmentor.predict(img_pil)

        # ── 3. Depth ──────────────────────────────────────────────────────
        print("Running depth estimation...")
        depth_norm = self.depth_est.predict(img_np)

        # ── 4. Placement ──────────────────────────────────────────────────
        print(f"Running placement engine ({intervention_type})...")
        engine    = PlacementEngine(seg_map, depth_norm, img_shape=(512, 1024))
        placement = engine.get_placement(intervention_type)

        if not placement["valid"]:
            print(f"Placement failed: {placement['reason']}")
            return {"valid": False, "reason": placement["reason"]}

        print(f"Placement valid: {placement['reason']}")

        # ── 5. LaMa inpainting ────────────────────────────────────────────
        if self.lama is not None:
            print("Running LaMa inpainting...")
            from scipy.ndimage import binary_dilation
            dilated  = binary_dilation(
                placement["mask"].astype(bool), iterations=6
            ).astype(np.uint8)
            clean_bg = self._lama_inpaint(img_np, dilated)
        else:
            clean_bg = img_np.copy()

        # ── 6. AccessNet synthesis ────────────────────────────────────────
        if self.generator is not None:
            print("Running AccessNet synthesis...")
            final_img = self._accessnet_synthesize(
                clean_bg, seg_map, depth_norm, placement)
        else:
            final_img = clean_bg

        # ── 7. Accessibility indicators ───────────────────────────────────
        all_results = {}
        for itype in ["crosswalk", "bench", "curb_ramp"]:
            all_results[itype] = engine.get_placement(itype)
        indicators = compute_indicators(all_results, seg_map)
        print("\n" + format_indicators(indicators))

        # ── 8. Save outputs ───────────────────────────────────────────────
        stem = os.path.splitext(os.path.basename(image_path))[0]

        before_path = os.path.join(output_dir, f"{stem}_before.png")
        after_path  = os.path.join(output_dir, f"{stem}_after.png")
        Image.fromarray(img_np).save(before_path)
        Image.fromarray(final_img).save(after_path)

        # Before/after comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        axes[0].imshow(img_np);    axes[0].set_title("Before", fontweight="bold", fontsize=14); axes[0].axis("off")
        axes[1].imshow(final_img); axes[1].set_title(f"After — {intervention_type}", fontweight="bold", fontsize=14); axes[1].axis("off")

        score = indicators["overall"]["score"]
        plt.suptitle(
            f"StreetSynth — Accessibility Score: {score}/3",
            fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, f"{stem}_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {comparison_path}")

        return {
            "valid":       True,
            "img_np":      img_np,
            "seg_map":     seg_map,
            "depth_norm":  depth_norm,
            "placement":   placement,
            "clean_bg":    clean_bg,
            "final_img":   final_img,
            "indicators":  indicators,
            "paths": {
                "before":     before_path,
                "after":      after_path,
                "comparison": comparison_path,
            }
        }


# ── CLI entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreetSynth Pipeline")
    parser.add_argument("--image",        required=True,   help="Input image path")
    parser.add_argument("--intervention", default="crosswalk",
                        choices=["crosswalk", "bench", "curb_ramp"])
    parser.add_argument("--checkpoint",   default=None,    help="AccessNet checkpoint path")
    parser.add_argument("--output_dir",   default="outputs")
    parser.add_argument("--lightweight",  action="store_true",
                        help="Skip LaMa (faster, for CPU)")
    args = parser.parse_args()

    pipeline = StreetSynthPipeline(
        checkpoint_path=args.checkpoint,
        lightweight=args.lightweight
    )
    result = pipeline.run(
        image_path=args.image,
        intervention_type=args.intervention,
        output_dir=args.output_dir
    )