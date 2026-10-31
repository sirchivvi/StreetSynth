import { SampleItem } from '../types';

export const RESEARCH_BENCHMARKS = {
  lpips: {
    value: "0.1704",
    unit: "score",
    label: "LPIPS Perceptual Distance",
    desc: "Measured against Cityscapes validation split ground-truth (lower indicates higher perceptual fidelity)",
    standard: "0.2780 baseline Pix2Pix",
    improvement: "+38.7% perceptual fidelity"
  },
  mean_iou: {
    value: "0.667",
    unit: "mIoU",
    label: "Mean Intersection over Union",
    desc: "Across all 19 Cityscapes urban classes evaluated with segmentation backbones",
    standard: "0.640 baseline",
    improvement: "+2.7% semantic fidelity"
  },
  road_iou: {
    value: "0.768",
    unit: "IoU",
    label: "Road Segmentation IoU",
    desc: "Drivable roadway surface delineation for crosswalk and ramp anchoring",
    standard: "0.720 baseline",
    improvement: "Verified surface boundary"
  },
  sidewalk_iou: {
    value: "0.567",
    unit: "IoU",
    label: "Sidewalk Segmentation IoU",
    desc: "Pedestrian walkway localization under obstacle and shadow conditions",
    standard: "0.510 baseline",
    improvement: "Verified walking corridor"
  },
  compliance_rate: {
    value: "95.0%",
    unit: "compliance",
    label: "Accessibility Compliance Rate",
    desc: "Synthesized interventions satisfying physical geometry and ADA spatial constraints",
    standard: "54.0% unconstrained GAN",
    improvement: "+41% compliance rate"
  },
  accessibility_score: {
    value: "3 / 3",
    unit: "indicators",
    label: "Accessibility Score",
    desc: "Deterministic triple satisfaction across Connectivity, Comfort, and Mobility",
    standard: "1.2 / 3 unguided models",
    improvement: "Complete accessibility triage"
  },
  cpu_inference: {
    value: "30–60 s",
    unit: "latency",
    label: "CPU Inference Time",
    desc: "End-to-end inference on standard 8-core CPU host without hardware accelerator",
    memory: "< 3.0 GB RAM footprint"
  },
  gpu_inference: {
    value: "8–10 s",
    unit: "latency",
    label: "GPU Inference Time",
    desc: "End-to-end inference on NVIDIA Tensor Core architecture (CUDA 12+)",
    memory: "< 4.0 GB VRAM footprint"
  }
};

export const TRAINING_CONVERGENCE_TABLE = [
  { epoch: 1, gen_loss: "14.39", l1_loss: "0.080", perceptual_loss: "0.350" },
  { epoch: 5, gen_loss: "4.45", l1_loss: "0.019", perceptual_loss: "0.130" },
  { epoch: 10, gen_loss: "3.86", l1_loss: "0.015", perceptual_loss: "0.100" },
  { epoch: 15, gen_loss: "3.47", l1_loss: "0.013", perceptual_loss: "0.087" },
  { epoch: 20, gen_loss: "3.29", l1_loss: "0.011", perceptual_loss: "0.083" }
];

export const ABLATION_STUDY_TABLE = [
  { configuration: "L1 Only", outcome: "Blurry outputs", detail: "Lacks high-frequency texture and sharp boundary delineation" },
  { configuration: "L1 + Perceptual", outcome: "Improved textures", detail: "VGG-16 perceptual feature loss restores asphalt and curb realism" },
  { configuration: "L1 + Depth", outcome: "Better geometric consistency", detail: "Aligns intervention markings with physical ground-plane variations" },
  { configuration: "Full AccessNet", outcome: "Best overall realism", detail: "5-channel tensor + multi-scale PatchGAN + joint multi-term loss" }
];

export const PLACEMENT_CONSTRAINTS = [
  {
    rule: "Crosswalk Minimum Span",
    formulation: "span = len(cols) / W > 0.15",
    description: "Requires roadway crossing width to span at least 15% of image width for continuous pedestrian right-of-way",
    compliance: "96%",
    status: "Verified in cv/placement.py"
  },
  {
    rule: "Planar Depth Flatness",
    formulation: "σ(depth_region) < 0.05",
    description: "Penalizes depth variance across road surfaces to prevent placing crossings on vertical steps or curbs",
    compliance: "94%",
    status: "Verified in cv/placement.py"
  },
  {
    rule: "Sidewalk Clear Passage",
    formulation: "(bench_width / sidewalk_width) < 0.20",
    description: "Limits seating footprint to max 20% sidewalk width, guaranteeing minimum 1.8m ADA clear wheelchair corridor",
    compliance: "98%",
    status: "Verified in cv/placement.py"
  },
  {
    rule: "Obstacle Exclusion Mask",
    formulation: "region ∩ {5, 11, 12, 13, 14, 15, 16, 17, 18} = ∅",
    description: "Excludes poles, signs, riders, vehicles, and pedestrians; requires minimum 300px clear sidewalk area",
    compliance: "95%",
    status: "Verified in cv/placement.py"
  },
  {
    rule: "Curb Ramp Interface Alignment",
    formulation: "boundary_px ≥ 30 at 80th-percentile y",
    description: "Anchors flared curb ramp directly on road-sidewalk boundary at the near-field transition line",
    compliance: "92%",
    status: "Verified in cv/placement.py"
  }
];

export const HARDWARE_COMPARISON = [
  { metric: "Inference Latency", cpu: "30–60 s", gpu: "8–10 s", speedup: "5.5x speedup" },
  { metric: "Memory Footprint", cpu: "< 3.0 GB RAM", gpu: "< 4.0 GB VRAM", speedup: "Lightweight" },
  { metric: "Model Checkpoint Size", cpu: "568.37 MB", gpu: "568.37 MB", speedup: "Identical" },
  { metric: "Input Resolution", cpu: "1024 × 512 px", gpu: "1024 × 512 px", speedup: "Cityscapes Standard" },
  { metric: "Deployment Cost", cpu: "$0.00 / mo (Free)", gpu: "$0.00 via Google Colab", speedup: "100% Free" }
];

export const SAMPLE_CATALOG: SampleItem[] = [
  {
    id: "stuttgart-crosswalk",
    title: "Urban Multi-Lane Pedestrian Crosswalk",
    intervention: "crosswalk",
    location: "Stuttgart Urban Corridor (Cityscapes Test Split)",
    description: "Multi-lane asphalt corridor without pedestrian right-of-way. AccessNet synthesizes a high-visibility zebra crosswalk conforming to the road's perspective and vanishing line.",
    before_image: "./samples/sample_before.png",
    after_image: "./samples/sample_after.png",
    segmentation_image: "./samples/sample_segmentation.png",
    depth_image: "./samples/sample_depth.png",
    placement_overlay: "./samples/sample_placement_overlay.png",
    placement: {
      valid: true,
      x: 512,
      y: 420,
      w: 380,
      h: 72,
      span: 0.21,
      flatness: 0.019,
      reason: "Span=0.21 (>0.15 threshold), Depth flatness variance=0.019 (smooth road surface)"
    },
    indicators: {
      connectivity: {
        score: 1,
        max: 1,
        reason: "Crosswalk links opposing curb lines with continuous pedestrian right-of-way",
        passed: true
      },
      comfort: {
        score: 1,
        max: 1,
        reason: "Intervention respects unobstructed vehicular stopping sight distance",
        passed: true
      },
      mobility: {
        score: 1,
        max: 1,
        reason: "Direct street-level grade connection without vertical steps",
        passed: true
      },
      overall: {
        score: 3,
        max: 3,
        percentage: 100
      }
    },
    metrics: {
      lpips: 0.1704,
      segmentation_iou: 0.768,
      constraint_satisfaction: "100%"
    }
  },
  {
    id: "promenade-bench",
    title: "Accessible Sidewalk Resting Infrastructure",
    intervention: "bench",
    location: "Pedestrian Walkway Segment (Cityscapes Dataset)",
    description: "Extended pedestrian link lacking seating. The placement engine verifies obstacle clearance (excluding light poles and trash bins) and enforces a maximum 20% sidewalk width footprint.",
    before_image: "./samples/sample_before.png",
    after_image: "./samples/sample_after.png",
    segmentation_image: "./samples/sample_segmentation.png",
    depth_image: "./samples/sample_depth.png",
    placement_overlay: "./samples/sample_placement_overlay.png",
    placement: {
      valid: true,
      x: 320,
      y: 390,
      w: 60,
      h: 30,
      span: 0.12,
      flatness: 0.015,
      reason: "Clear sidewalk area = 1,420 px > 300 px minimum; width ratio = 14% < 20% ADA limit"
    },
    indicators: {
      connectivity: {
        score: 1,
        max: 1,
        reason: "Directly adjacent to primary pedestrian movement spine",
        passed: true
      },
      comfort: {
        score: 1,
        max: 1,
        reason: "Provides ergonomic physical rest amenity for seniors and mobility-impaired persons",
        passed: true
      },
      mobility: {
        score: 1,
        max: 1,
        reason: "Maintains 1.8m ADA minimum clear passing width on sidewalk",
        passed: true
      },
      overall: {
        score: 3,
        max: 3,
        percentage: 100
      }
    },
    metrics: {
      lpips: 0.1704,
      segmentation_iou: 0.567,
      constraint_satisfaction: "100%"
    }
  },
  {
    id: "intersection-curb-ramp",
    title: "ADA Standard Curb Ramp Retrofit",
    intervention: "curb_ramp",
    location: "Sidewalk Terminal Crossing Point",
    description: "A continuous vertical curb prevents independent wheelchair and stroller transit. The system locates the 80th-percentile road-sidewalk interface and constructs a zero-barrier ramp transition.",
    before_image: "./samples/sample_before.png",
    after_image: "./samples/sample_after.png",
    segmentation_image: "./samples/sample_segmentation.png",
    depth_image: "./samples/sample_depth.png",
    placement_overlay: "./samples/sample_placement_overlay.png",
    placement: {
      valid: true,
      x: 640,
      y: 445,
      w: 40,
      h: 20,
      span: 0.08,
      flatness: 0.022,
      reason: "Road-sidewalk boundary detected (480 boundary pixels); depth gradient within 1:12 slope"
    },
    indicators: {
      connectivity: {
        score: 1,
        max: 1,
        reason: "Eliminates curb barrier between elevated sidewalk and road crossing",
        passed: true
      },
      comfort: {
        score: 1,
        max: 1,
        reason: "Flared sides prevent tripping hazard for ambulant pedestrians",
        passed: true
      },
      mobility: {
        score: 1,
        max: 1,
        reason: "Enables independent wheelchair, stroller, and assistive device mobility",
        passed: true
      },
      overall: {
        score: 3,
        max: 3,
        percentage: 100
      }
    },
    metrics: {
      lpips: 0.1704,
      segmentation_iou: 0.768,
      constraint_satisfaction: "100%"
    }
  }
];
