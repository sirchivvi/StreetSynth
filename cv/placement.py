import numpy as np
from scipy import ndimage

class PlacementEngine:
    def __init__(self, seg_map, depth_norm, img_shape=(512, 1024)):
        self.seg           = seg_map
        self.depth         = depth_norm
        self.H, self.W     = img_shape
        self.road_mask     = (seg_map == 0).astype(np.uint8)
        self.sidewalk_mask = (seg_map == 1).astype(np.uint8)

    def _depth_flatness(self, mask):
        vals = self.depth[mask > 0]
        return float(np.std(vals)) if len(vals) > 10 else 1.0

    def _road_sidewalk_boundary(self):
        dilated_sw = ndimage.binary_dilation(self.sidewalk_mask, iterations=5)
        return (dilated_sw & self.road_mask).astype(np.uint8)

    def find_crosswalk(self):
        boundary = self._road_sidewalk_boundary()
        labeled, num = ndimage.label(boundary)
        best, best_score = None, -1
        for rid in range(1, num + 1):
            region = (labeled == rid)
            cols = np.where(region.any(axis=0))[0]
            rows = np.where(region.any(axis=1))[0]
            if len(cols) < 10: continue
            span  = len(cols) / self.W
            flat  = self._depth_flatness(region)
            score = span - flat * 0.5
            if span > 0.15 and score > best_score:
                best_score = score
                best = (int(cols.mean()), int(rows.mean()),
                        int(len(cols)), max(30, int(len(rows)*2)),
                        f"span={span:.2f} flatness={flat:.3f}")
        return best

    def find_bench(self):
        obstacle_ids = {5, 11, 12, 13, 14, 15, 16, 17, 18}
        obs_mask = np.isin(self.seg, list(obstacle_ids))
        valid_sw = (self.sidewalk_mask & ~obs_mask).astype(np.uint8)
        if valid_sw.sum() < 300:
            return None
        labeled, num = ndimage.label(valid_sw)
        sizes   = ndimage.sum(valid_sw, labeled, range(1, num + 1))
        best_id = int(np.argmax(sizes)) + 1
        region  = (labeled == best_id)
        rows = np.where(region.any(axis=1))[0]
        cols = np.where(region.any(axis=0))[0]
        flat = self._depth_flatness(region)
        sw_width = self.sidewalk_mask.any(axis=0).sum()
        if sw_width > 0 and (60 / sw_width) > 0.20:
            return None
        return (int(cols.mean()), int(rows.mean()), 60, 30,
                f"sw_area={region.sum()} flatness={flat:.3f}")

    def find_curb_ramp(self):
        boundary = self._road_sidewalk_boundary()
        if boundary.sum() < 30:
            return None
        rows, cols = np.where(boundary > 0)
        threshold  = np.percentile(rows, 80)
        near_mask  = rows >= threshold
        y_c  = int(rows[near_mask].mean())
        x_c  = int(cols[near_mask].mean())
        flat = self._depth_flatness(boundary)
        return (x_c, y_c, 40, 20,
                f"boundary_px={boundary.sum()} flatness={flat:.3f}")

    def get_placement(self, itype):
        fn = {"crosswalk": self.find_crosswalk,
              "bench":     self.find_bench,
              "curb_ramp": self.find_curb_ramp}
        if itype not in fn:
            raise ValueError(f"Unknown intervention: {itype}")
        result = fn[itype]()
        if result is None:
            return {"valid": False, "reason": f"No valid location for {itype}"}
        x, y, w, h, reason = result
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        x1, x2 = max(0, x-w//2), min(self.W, x+w//2)
        y1, y2 = max(0, y-h//2), min(self.H, y+h//2)
        mask[y1:y2, x1:x2] = 1
        return {"valid": True, "x":x, "y":y, "w":w, "h":h,
                "mask":mask, "reason":reason}