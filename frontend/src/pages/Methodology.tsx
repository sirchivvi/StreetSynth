import React, { useState } from 'react';
import { Zap } from 'lucide-react';

export const Methodology: React.FC = () => {
  const [selectedStage, setSelectedStage] = useState<number>(4);

  const stages = [
    {
      num: "01",
      title: "Street Photo Input",
      tech: "RGB Input Tensor (1024 × 512 × 3)",
      desc: "Standard perspective street imagery normalized into zero-mean, unit-variance color space.",
      input: "Raw camera photo (JPEG/PNG)",
      output: "Normalized tensor X ∈ ℝ^(3 × 512 × 1024)",
      detail: "Captures natural focal depth, ambient illumination, vanishing lines, and roadway asphalt textures."
    },
    {
      num: "02",
      title: "Semantic Segmentation",
      tech: "DeepLabV3+ (Research Baseline) & SegFormer-B5 (Active Pipeline)",
      desc: "DeepLabV3+ semantic segmentation architecture, implemented with SegFormer-B5 Cityscapes 1024×1024 transformer backbone for dense 19-class boundary delineation.",
      input: "Normalized RGB Image",
      output: "Discrete semantic mask S ∈ {0,...,18}^(512 × 1024)",
      detail: "Isolates class 0 (Road), class 1 (Sidewalk), and classes 11-18 (Vehicles, Poles, Pedestrians) for barrier-free zoning."
    },
    {
      num: "03",
      title: "Monocular Depth Estimation",
      tech: "MiDaS (Research Baseline) & DPT-Large (Active Pipeline)",
      desc: "MiDaS relative inverse depth estimation architecture, implemented with Intel DPT-Large (MiDaS v3 vision transformer) for dense relative depth field extraction.",
      input: "RGB Image",
      output: "Continuous normalized depth D ∈ [0, 1]^(512 × 1024)",
      detail: "Computes 3D ground plane slopes, road curvature, and planar flatness variance σ to prevent warped perspective."
    },
    {
      num: "04",
      title: "Accessibility Placement Engine",
      tech: "Deterministic Heuristic Optimization (cv/placement.py)",
      desc: "Mathematical rule validator verifying spatial feasibility, obstacle clearance, and ADA guidelines.",
      input: "Semantic Mask S + Depth Map D",
      output: "Placement Bounding Box (x, y, w, h) + Mask M ∈ {0, 1}",
      detail: "Validates minimum crosswalk span (>15% width), sidewalk clear passing corridor (1.8m), and curb ramp elevation transitions."
    },
    {
      num: "05",
      title: "AccessNet Conditional GAN & LaMa",
      tech: "5-Channel U-Net + MultiScale PatchGAN + LaMa",
      desc: "Inpaints existing obstructions via LaMa and generates photorealistic accessibility interventions conditioned on depth & semantics.",
      input: "5-Channel Concatenation [RGB(3) + Seg(1) + Depth(1)]",
      output: "Synthesized Output Image Y ∈ ℝ^(3 × 512 × 1024)",
      detail: "Jointly optimized under adversarial, VGG16 perceptual L1, and depth-consistency gradient penalty losses."
    }
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-16">
      
      {/* Page Header */}
      <div className="space-y-3 pb-6 border-b border-[#26322C]">
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#10B981]"></span>
          <span className="text-xs font-mono uppercase tracking-wider text-[#10B981]">
            Technical Architecture
          </span>
        </div>
        <h1 className="text-3xl font-bold text-[#F3F5F4] tracking-tight">
          System Methodology &amp; Mathematical Formulation
        </h1>
        <p className="text-sm text-[#9CA8A0] max-w-3xl leading-relaxed">
          Comprehensive breakdown of the 5-stage synthesis pipeline, multi-modal 5-channel tensor formulation, placement constraint algorithms, and joint loss optimization functions.
        </p>
      </div>


      {/* ── ARCHITECTURAL DISTINCTION BANNER ───────────────────────────────── */}
      <div className="bg-[#181F1B] border border-[#26322C] rounded-xl p-5 font-mono text-xs text-[#9CA8A0] space-y-2">
        <div className="flex items-center space-x-2 text-[#F3F5F4]">
          <span className="w-2 h-2 rounded-full bg-[#10B981]" />
          <span className="font-bold text-sm">Model Architecture Lineage</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          <div className="p-3 rounded bg-[#141A17] border border-[#26322C] space-y-1">
            <span className="text-[#34D399] font-bold">1. Foundational Research Architecture:</span>
            <p className="text-[11px] text-[#6E7C73] leading-relaxed">
              Formulated on <strong>DeepLabV3+</strong> for semantic parsing, <strong>MiDaS</strong> for monocular depth, deterministic rule-based placement engine, <strong>LaMa</strong> inpainting, and the custom <strong>AccessNet</strong> conditional GAN.
            </p>
          </div>
          <div className="p-3 rounded bg-[#141A17] border border-[#26322C] space-y-1">
            <span className="text-[#34D399] font-bold">2. High-Resolution Transformer Pipeline:</span>
            <p className="text-[11px] text-[#6E7C73] leading-relaxed">
              The active codebase employs <strong>SegFormer-B5</strong> (fine-tuned on Cityscapes 1024×1024) and <strong>DPT-Large (MiDaS v3)</strong> for high-resolution edge preservation in the inference pipeline.
            </p>
          </div>
        </div>
      </div>


      {/* ── INTERACTIVE 5-STAGE PIPELINE ARCHITECTURE FLOW ─────────────────── */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-mono uppercase tracking-wider text-[#9CA8A0]">
            Interactive Pipeline Flowchart (Click any stage)
          </h2>
          <span className="text-xs font-mono text-[#34D399]">
            Active: Stage {stages[selectedStage - 1].num} — {stages[selectedStage - 1].title}
          </span>
        </div>

        {/* Pipeline Stage Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {stages.map((stage, i) => {
            const isSelected = selectedStage === i + 1;
            return (
              <button
                key={stage.num}
                onClick={() => setSelectedStage(i + 1)}
                className={`p-4 rounded-lg text-left border transition-all ${
                  isSelected
                    ? 'bg-[#1D2521] border-[#10B981] shadow-subtle-emerald'
                    : 'bg-[#181F1B] border-[#26322C] hover:border-[#34423A]'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-mono mb-2">
                  <span className={`font-bold ${isSelected ? 'text-[#34D399]' : 'text-[#6E7C73]'}`}>
                    STAGE {stage.num}
                  </span>
                  {isSelected && <Zap className="w-3.5 h-3.5 text-[#34D399]" />}
                </div>
                <div className="text-xs font-semibold text-[#F3F5F4] truncate">
                  {stage.title}
                </div>
                <div className="text-[11px] font-mono text-[#9CA8A0] mt-1 truncate">
                  {stage.tech}
                </div>
              </button>
            );
          })}
        </div>

        {/* Selected Stage Detail Card */}
        <div className="bg-[#181F1B] border border-[#26322C] rounded-xl p-6 sm:p-8 grid grid-cols-1 md:grid-cols-12 gap-8">
          <div className="md:col-span-8 space-y-4">
            <div className="flex items-center space-x-3">
              <span className="w-8 h-8 rounded bg-[#10B981] text-[#101412] font-mono font-bold flex items-center justify-center text-sm">
                {stages[selectedStage - 1].num}
              </span>
              <div>
                <h3 className="text-lg font-bold text-[#F3F5F4]">
                  {stages[selectedStage - 1].title}
                </h3>
                <span className="text-xs font-mono text-[#34D399]">
                  {stages[selectedStage - 1].tech}
                </span>
              </div>
            </div>

            <p className="text-sm text-[#9CA8A0] leading-relaxed">
              {stages[selectedStage - 1].desc}
            </p>

            <p className="text-xs text-[#F3F5F4] bg-[#141A17] p-3 rounded-lg border border-[#26322C] leading-relaxed">
              <strong className="text-[#34D399] font-mono">Implementation Detail: </strong>
              {stages[selectedStage - 1].detail}
            </p>
          </div>

          <div className="md:col-span-4 bg-[#101412] p-5 rounded-lg border border-[#26322C] font-mono text-xs space-y-3">
            <div className="text-[#6E7C73] uppercase tracking-wider text-[10px]">Data Flow</div>
            
            <div className="space-y-1">
              <div className="text-[#9CA8A0]">Tensor Input:</div>
              <div className="text-[#F3F5F4] bg-[#181F1B] p-2 rounded border border-[#26322C] text-[11px] break-all">
                {stages[selectedStage - 1].input}
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[#9CA8A0]">Tensor Output:</div>
              <div className="text-[#34D399] bg-[#181F1B] p-2 rounded border border-[#26322C] text-[11px] break-all">
                {stages[selectedStage - 1].output}
              </div>
            </div>
          </div>
        </div>

      </div>


      {/* ── ACCESSNET 5-CHANNEL TENSOR ARCHITECTURE ─────────────────────────── */}
      <div className="bg-[#181F1B] border border-[#26322C] rounded-xl p-8 space-y-6">
        <div>
          <span className="text-xs font-mono uppercase tracking-wider text-[#10B981]">
            Neural Network Architecture
          </span>
          <h2 className="text-2xl font-bold text-[#F3F5F4] tracking-tight mt-1">
            AccessNet 5-Channel U-Net Generator
          </h2>
          <p className="text-sm text-[#9CA8A0] mt-1 leading-relaxed max-w-3xl">
            AccessNet reformulates conditional street synthesis by concatenating the cleaned 3-channel RGB image with the 1-channel normalized semantic mask and the 1-channel continuous depth map into a unified 5-channel tensor.
          </p>
        </div>

        {/* 5-Channel Diagram */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 font-mono text-xs text-center">
          <div className="p-4 rounded bg-[#101412] border border-[#26322C]">
            <div className="text-red-400 font-bold">Ch 1: Red (R)</div>
            <div className="text-[10px] text-[#6E7C73] mt-1">RGB Visual Color</div>
          </div>
          <div className="p-4 rounded bg-[#101412] border border-[#26322C]">
            <div className="text-green-400 font-bold">Ch 2: Green (G)</div>
            <div className="text-[10px] text-[#6E7C73] mt-1">RGB Visual Color</div>
          </div>
          <div className="p-4 rounded bg-[#101412] border border-[#26322C]">
            <div className="text-blue-400 font-bold">Ch 3: Blue (B)</div>
            <div className="text-[10px] text-[#6E7C73] mt-1">RGB Visual Color</div>
          </div>
          <div className="p-4 rounded bg-[#1D2521] border border-[#10B981]/40">
            <div className="text-[#34D399] font-bold">Ch 4: Semantics</div>
            <div className="text-[10px] text-[#9CA8A0] mt-1">S / 19.0 Normalized</div>
          </div>
          <div className="p-4 rounded bg-[#1D2521] border border-[#10B981]/40">
            <div className="text-[#34D399] font-bold">Ch 5: Depth Field</div>
            <div className="text-[10px] text-[#9CA8A0] mt-1">Relative Inverse Depth</div>
          </div>
        </div>

        <div className="p-4 rounded bg-[#101412] border border-[#26322C] font-mono text-xs text-[#9CA8A0] space-y-2">
          <div className="text-[#F3F5F4] font-semibold">U-Net Topology &amp; Discriminator:</div>
          <div className="leading-relaxed text-[11px]">
            • <strong>Generator:</strong> 7 downsampling blocks with LeakyReLU and InstanceNorm + 6 upsampling blocks with transpose convolutions, skip concatenations, and 50% dropout in the bottleneck layers. Output activation is <code>Tanh()</code> mapped to [-1, 1].
          </div>
          <div className="leading-relaxed text-[11px]">
            • <strong>Discriminator:</strong> Multi-Scale PatchGAN with 2 scales (D1 full resolution, D2 downsampled by AvgPool2d). Evaluates 70×70 receptive field patches to verify both global street consistency and fine asphalt markings.
          </div>
        </div>
      </div>


      {/* ── LOSS FUNCTIONS & MATHEMATICAL FORMULATIONS ──────────────────────── */}
      <div className="space-y-6">
        <div>
          <span className="text-xs font-mono uppercase tracking-wider text-[#10B981]">
            Objective Functions
          </span>
          <h2 className="text-2xl font-bold text-[#F3F5F4] tracking-tight mt-1">
            Multi-Objective Loss Formulation
          </h2>
          <p className="text-sm text-[#9CA8A0] mt-1 leading-relaxed">
            AccessNet is trained using a composite loss function penalizing adversarial divergence, feature perceptual distance, and depth-geometry distortion.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          
          {/* Adversarial Loss */}
          <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-5 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[#34D399] font-semibold">1. Adversarial Loss</span>
              <span className="text-[#6E7C73]">cGAN</span>
            </div>
            <div className="p-3 rounded bg-[#101412] border border-[#26322C] font-mono text-xs text-[#F3F5F4] overflow-x-auto">
              L_cGAN = E[log D(x, y)] + E[log(1 - D(x, G(x)))]
            </div>
            <p className="text-xs text-[#9CA8A0] leading-relaxed">
              Enforces realistic textures and prevents blurred artifacts by forcing the multi-scale discriminator to distinguish real Cityscapes street patches from synthesized interventions.
            </p>
          </div>

          {/* Perceptual Loss */}
          <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-5 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[#34D399] font-semibold">2. Perceptual L1 Loss</span>
              <span className="text-[#6E7C73]">VGG-16</span>
            </div>
            <div className="p-3 rounded bg-[#101412] border border-[#26322C] font-mono text-xs text-[#F3F5F4] overflow-x-auto">
              L_perc = Σ || φ_l(y) - φ_l(G(x)) ||_1
            </div>
            <p className="text-xs text-[#9CA8A0] leading-relaxed">
              Extracts intermediate feature activations from VGG-16 layers (relu1_2, relu2_2, relu3_3) to preserve semantic urban structures and high-level structural integrity.
            </p>
          </div>

          {/* Depth Consistency Loss */}
          <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-5 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[#34D399] font-semibold">3. Depth Consistency Loss</span>
              <span className="text-[#6E7C73]">Edge-Aware</span>
            </div>
            <div className="p-3 rounded bg-[#101412] border border-[#26322C] font-mono text-xs text-[#F3F5F4] overflow-x-auto">
              L_depth = |∇y|·e^(-|∇d|) + |∇x|·e^(-|∇d|)
            </div>
            <p className="text-xs text-[#9CA8A0] leading-relaxed">
              Novel edge-aware penalty ensuring synthesized interventions align with physical surface gradients, preventing synthetic crosswalks or ramps from floating above asphalt.
            </p>
          </div>

        </div>
      </div>


      {/* ── PLACEMENT ENGINE RULE DEFINITIONS ───────────────────────────────── */}
      <div className="bg-[#181F1B] border border-[#26322C] rounded-xl p-8 space-y-6">
        <div>
          <span className="text-xs font-mono uppercase tracking-wider text-[#10B981]">
            Deterministic Algorithms
          </span>
          <h2 className="text-2xl font-bold text-[#F3F5F4] tracking-tight mt-1">
            Algorithmic Placement Constraints (cv/placement.py)
          </h2>
          <p className="text-sm text-[#9CA8A0] mt-1 leading-relaxed">
            The placement engine calculates geometric feasibility across three municipal infrastructure domains.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 font-mono text-xs">
          
          {/* Crosswalk Rule */}
          <div className="p-5 rounded-lg bg-[#141A17] border border-[#26322C] space-y-3">
            <div className="text-[#34D399] font-bold text-sm">Crosswalk Placement</div>
            <div className="text-[#9CA8A0] leading-relaxed text-[11px] space-y-2">
              <p>• <strong>Boundary Extraction:</strong> Dilates sidewalk mask by 5 iterations and intersects with road mask: <code>(dilated_sw &amp; road_mask)</code>.</p>
              <p>• <strong>Minimum Span:</strong> Enforces <code>span = len(cols) / W &gt; 0.15</code> (at least 15% image width).</p>
              <p>• <strong>Flatness Penalty:</strong> Calculates depth standard deviation across candidate region; scores via <code>score = span - 0.5 * σ_depth</code>.</p>
            </div>
          </div>

          {/* Bench Rule */}
          <div className="p-5 rounded-lg bg-[#141A17] border border-[#26322C] space-y-3">
            <div className="text-[#34D399] font-bold text-sm">Sidewalk Bench Placement</div>
            <div className="text-[#9CA8A0] leading-relaxed text-[11px] space-y-2">
              <p>• <strong>Obstacle Exclusion:</strong> Masks out building, pole, sign, rider, vehicle, and person IDs: <code>{'{5, 11, 12, 13, 14, 15, 16, 17, 18}'}</code>.</p>
              <p>• <strong>Footprint Limit:</strong> Enforces <code>(60px / sw_width) &lt; 0.20</code> to maintain 1.8m ADA minimum clear passage.</p>
              <p>• <strong>Area Constraint:</strong> Rejects candidate regions with unobstructed sidewalk area &lt; 300 pixels.</p>
            </div>
          </div>

          {/* Curb Ramp Rule */}
          <div className="p-5 rounded-lg bg-[#141A17] border border-[#26322C] space-y-3">
            <div className="text-[#34D399] font-bold text-sm">Curb Ramp Alignment</div>
            <div className="text-[#9CA8A0] leading-relaxed text-[11px] space-y-2">
              <p>• <strong>Boundary Interface:</strong> Requires at least 30 intersecting interface pixels between roadway and curb.</p>
              <p>• <strong>80th-Percentile Depth:</strong> Positions ramp anchor at <code>threshold = np.percentile(rows, 80)</code> for near-field pedestrian entry.</p>
              <p>• <strong>Tactile Flaring:</strong> Standardized 40×20 px aspect ratio matching standard municipal curb transition slope.</p>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
};
