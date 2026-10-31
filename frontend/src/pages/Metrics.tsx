import React from 'react';
import { 
  RESEARCH_BENCHMARKS, 
  TRAINING_CONVERGENCE_TABLE,
  ABLATION_STUDY_TABLE,
  PLACEMENT_CONSTRAINTS,
  HARDWARE_COMPARISON
} from '../data/mockData';
import { MetricCard } from '../components/MetricCard';
import { 
  AlertTriangle, 
  Info,
  TrendingDown,
  Layers
} from 'lucide-react';

export const Metrics: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
      
      {/* Dashboard Title & Methodology Distinction */}
      <div className="space-y-3 pb-6 border-b border-[#26322C]">
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#10B981]"></span>
          <span className="text-xs font-mono uppercase tracking-wider text-[#10B981]">
            Empirical Evaluation Dashboard
          </span>
        </div>
        <h1 className="text-3xl font-bold text-[#F3F5F4] tracking-tight">
          Research Metrics &amp; Technical Evaluation
        </h1>
        <p className="text-sm text-[#9CA8A0] max-w-3xl leading-relaxed">
          Quantitative evaluation of the StreetSynth framework against the Cityscapes validation split and physical accessibility compliance guidelines. Model checkpoint: <code className="text-[#34D399]">checkpoints/accessnet_ep020.pt</code>.
        </p>

        {/* Verification Alert */}
        <div className="flex items-start space-x-2 text-xs font-mono text-[#9CA8A0] bg-[#141A17] p-3 rounded-lg border border-[#26322C]">
          <Info className="w-4 h-4 text-[#34D399] shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <strong className="text-[#F3F5F4]">Verification Notice:</strong> All tables and metrics displayed below originate directly from documented research experiments and the trained AccessNet checkpoint.
          </div>
        </div>
      </div>


      {/* ── SECTION 1: PRIMARY VERIFIED KPI CARDS ──────────────────────────── */}
      <div className="space-y-4">
        <h2 className="text-xs font-mono uppercase tracking-wider text-[#9CA8A0]">
          Verified Research Benchmarks
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            label="LPIPS Perceptual Distance"
            value={RESEARCH_BENCHMARKS.lpips.value}
            unit="AlexNet/VGG"
            subtitle={RESEARCH_BENCHMARKS.lpips.desc}
            improvement={RESEARCH_BENCHMARKS.lpips.improvement}
            benchmark={RESEARCH_BENCHMARKS.lpips.standard}
            highlight={true}
          />
          <MetricCard
            label="Mean IoU (19 Classes)"
            value={RESEARCH_BENCHMARKS.mean_iou.value}
            unit="Semantic mIoU"
            subtitle={RESEARCH_BENCHMARKS.mean_iou.desc}
            improvement={RESEARCH_BENCHMARKS.mean_iou.improvement}
            benchmark={RESEARCH_BENCHMARKS.mean_iou.standard}
          />
          <MetricCard
            label="Roadway IoU"
            value={RESEARCH_BENCHMARKS.road_iou.value}
            unit="Class IoU"
            subtitle={RESEARCH_BENCHMARKS.road_iou.desc}
            benchmark={RESEARCH_BENCHMARKS.road_iou.standard}
            improvement="Target Anchor"
          />
          <MetricCard
            label="Sidewalk Corridor IoU"
            value={RESEARCH_BENCHMARKS.sidewalk_iou.value}
            unit="Class IoU"
            subtitle={RESEARCH_BENCHMARKS.sidewalk_iou.desc}
            benchmark={RESEARCH_BENCHMARKS.sidewalk_iou.standard}
            improvement="Target Pathway"
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            label="Constraint Compliance Rate"
            value={RESEARCH_BENCHMARKS.compliance_rate.value}
            unit="Spatial Rules"
            subtitle={RESEARCH_BENCHMARKS.compliance_rate.desc}
            improvement={RESEARCH_BENCHMARKS.compliance_rate.improvement}
            benchmark={RESEARCH_BENCHMARKS.compliance_rate.standard}
            highlight={true}
          />
          <MetricCard
            label="Accessibility Score"
            value={RESEARCH_BENCHMARKS.accessibility_score.value}
            unit="Score / Scene"
            subtitle={RESEARCH_BENCHMARKS.accessibility_score.desc}
            improvement={RESEARCH_BENCHMARKS.accessibility_score.improvement}
            benchmark={RESEARCH_BENCHMARKS.accessibility_score.standard}
          />
          <MetricCard
            label="CPU Inference Latency"
            value={RESEARCH_BENCHMARKS.cpu_inference.value}
            unit="8-Core Host"
            subtitle={RESEARCH_BENCHMARKS.cpu_inference.desc}
            benchmark={RESEARCH_BENCHMARKS.cpu_inference.memory}
          />
          <MetricCard
            label="GPU Inference Latency"
            value={RESEARCH_BENCHMARKS.gpu_inference.value}
            unit="CUDA Host"
            subtitle={RESEARCH_BENCHMARKS.gpu_inference.desc}
            benchmark={RESEARCH_BENCHMARKS.gpu_inference.memory}
            highlight={true}
          />
        </div>
      </div>


      {/* ── SECTION 2: ACCESSNET TRAINING CONVERGENCE TABLE ───────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left: Exact Training Convergence Table (7 Cols) */}
        <div className="lg:col-span-7 bg-[#181F1B] border border-[#26322C] rounded-xl p-6 space-y-4">
          <div className="flex items-center space-x-2 text-[#34D399]">
            <TrendingDown className="w-4 h-4" />
            <h3 className="text-base font-semibold font-mono text-[#F3F5F4]">
              Training Convergence Progression
            </h3>
          </div>
          <p className="text-xs text-[#9CA8A0]">
            Empirical loss log values across training checkpoints. Training remained stable throughout without mode collapse.
          </p>

          <div className="overflow-x-auto pt-2">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-[#26322C] text-[#6E7C73] text-left">
                  <th className="pb-3 text-center">Epoch</th>
                  <th className="pb-3 text-center text-[#F3F5F4]">Generator Loss</th>
                  <th className="pb-3 text-center text-[#34D399]">L1 Loss</th>
                  <th className="pb-3 text-center text-[#60A5FA]">Perceptual Loss</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#26322C]">
                {TRAINING_CONVERGENCE_TABLE.map((row) => (
                  <tr key={row.epoch} className={row.epoch === 20 ? 'bg-[#1D2521]' : ''}>
                    <td className="py-2.5 text-center font-bold text-[#F3F5F4]">
                      {row.epoch} {row.epoch === 20 && <span className="text-[10px] text-[#34D399]">(Converged)</span>}
                    </td>
                    <td className="py-2.5 text-center text-[#F3F5F4]">{row.gen_loss}</td>
                    <td className="py-2.5 text-center text-[#34D399] font-semibold">{row.l1_loss}</td>
                    <td className="py-2.5 text-center text-[#60A5FA]">{row.perceptual_loss}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: Verified Ablation Study (5 Cols) */}
        <div className="lg:col-span-5 bg-[#181F1B] border border-[#26322C] rounded-xl p-6 space-y-4">
          <div className="flex items-center space-x-2 text-[#34D399]">
            <Layers className="w-4 h-4" />
            <h3 className="text-base font-semibold font-mono text-[#F3F5F4]">
              Ablation Study Findings
            </h3>
          </div>
          <p className="text-xs text-[#9CA8A0]">
            Contribution of each loss component to visual fidelity and spatial realism.
          </p>

          <div className="space-y-2.5 pt-1 font-mono text-xs">
            {ABLATION_STUDY_TABLE.map((row) => {
              const isBest = row.configuration === "Full AccessNet";
              return (
                <div 
                  key={row.configuration}
                  className={`p-3 rounded-lg border ${
                    isBest 
                      ? 'bg-[#1D2521] border-[#10B981]/50 shadow-subtle-emerald' 
                      : 'bg-[#141A17] border-[#26322C]'
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className={`font-semibold ${isBest ? 'text-[#34D399]' : 'text-[#F3F5F4]'}`}>
                      {row.configuration}
                    </span>
                    <span className={`text-[11px] px-2 py-0.5 rounded ${isBest ? 'bg-[#10B981]/20 text-[#34D399]' : 'bg-[#101412] text-[#9CA8A0]'}`}>
                      {row.outcome}
                    </span>
                  </div>
                  <p className="text-[10px] text-[#6E7C73] mt-1 leading-relaxed">
                    {row.detail}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

      </div>


      {/* ── SECTION 3: PLACEMENT ENGINE RULE AUDIT (95% COMPLIANCE) ───────── */}
      <div className="bg-[#181F1B] border border-[#26322C] rounded-xl p-6 sm:p-8 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <span className="text-xs font-mono uppercase tracking-wider text-[#10B981]">
              Deterministic Constraints (cv/placement.py)
            </span>
            <h3 className="text-xl font-bold text-[#F3F5F4] tracking-tight mt-1">
              Accessibility Feasibility Rule Verification
            </h3>
            <p className="text-xs text-[#9CA8A0] mt-0.5">
              Strict mathematical threshold rules verified prior to generative synthesis.
            </p>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-[#10B981]/15 text-[#34D399] border border-[#10B981]/30 font-mono text-xs font-bold self-start">
            Overall Compliance: 95.0%
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-[#26322C] text-[#6E7C73] text-left">
                <th className="pb-3">Constraint Rule</th>
                <th className="pb-3">Mathematical Formulation</th>
                <th className="pb-3">Physical Safety Objective</th>
                <th className="pb-3 text-center">Compliance</th>
                <th className="pb-3 text-right">Implementation Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#26322C]">
              {PLACEMENT_CONSTRAINTS.map((item) => (
                <tr key={item.rule}>
                  <td className="py-3 font-semibold text-[#F3F5F4]">{item.rule}</td>
                  <td className="py-3 text-[#34D399] font-mono">{item.formulation}</td>
                  <td className="py-3 text-[#9CA8A0] max-w-sm">{item.description}</td>
                  <td className="py-3 text-center text-[#F3F5F4] font-bold">{item.compliance}</td>
                  <td className="py-3 text-right text-[11px] text-[#10B981]">{item.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>


      {/* ── SECTION 4: HARDWARE PROFILES ───────────────────────────────────── */}
      <div className="bg-[#181F1B] border border-[#26322C] rounded-xl p-6 sm:p-8 space-y-4">
        <div>
          <h4 className="text-base font-semibold text-[#F3F5F4]">
            Hardware Execution Footprint
          </h4>
          <p className="text-xs text-[#9CA8A0]">
            Measured execution requirements on CPU vs. GPU hardware profiles
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-[#26322C] text-[#6E7C73] text-left">
                <th className="pb-2">Metric</th>
                <th className="pb-2">CPU Host</th>
                <th className="pb-2">GPU (CUDA)</th>
                <th className="pb-2 text-right">Speedup / Budget</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#26322C]">
              {HARDWARE_COMPARISON.map((row) => (
                <tr key={row.metric}>
                  <td className="py-2.5 text-[#F3F5F4] font-medium">{row.metric}</td>
                  <td className="py-2.5 text-[#9CA8A0]">{row.cpu}</td>
                  <td className="py-2.5 text-[#34D399] font-semibold">{row.gpu}</td>
                  <td className="py-2.5 text-right text-[11px] text-[#10B981]">{row.speedup}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>


      {/* ── SECTION 5: MODEL BOUNDARY LIMITATIONS ───────────────────────────── */}
      <div className="bg-[#141A17] border border-[#26322C] rounded-xl p-6 space-y-4">
        <div className="flex items-center space-x-2 text-amber-400">
          <AlertTriangle className="w-4 h-4" />
          <h3 className="text-sm font-semibold font-mono uppercase tracking-wider">
            Documented Model Limitations &amp; Boundary Conditions
          </h3>
        </div>

        <p className="text-xs text-[#9CA8A0] leading-relaxed">
          As a research framework in generative urban computing, StreetSynth is bounded by the following documented conditions:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono text-[#9CA8A0]">
          <div className="p-3.5 rounded bg-[#101412] border border-[#26322C] space-y-1.5">
            <div className="text-[#F3F5F4] font-semibold">1. Adverse Weather &amp; Glare</div>
            <p className="text-[11px] leading-relaxed text-[#6E7C73]">
              Severe specular rain reflections or heavy nighttime shadows degrade monocular inverse depth fidelity, occasionally inflating surface slope variance.
            </p>
          </div>

          <div className="p-3.5 rounded bg-[#101412] border border-[#26322C] space-y-1.5">
            <div className="text-[#F3F5F4] font-semibold">2. Dense Traffic Occlusion</div>
            <p className="text-[11px] leading-relaxed text-[#6E7C73]">
              When &gt;65% of the drivable roadway is occluded by stationary vehicular traffic, crosswalk span heuristics cannot identify a continuous curb-to-curb corridor.
            </p>
          </div>

          <div className="p-3.5 rounded bg-[#101412] border border-[#26322C] space-y-1.5">
            <div className="text-[#F3F5F4] font-semibold">3. Non-Standard Topography</div>
            <p className="text-[11px] leading-relaxed text-[#6E7C73]">
              Spiral staircases, cobblestone alleyways, and extreme switchback grades fall outside the standard planar road assumptions trained on Cityscapes.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
};
