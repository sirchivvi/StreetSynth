import React from 'react';
import { PageId } from '../types';
import { BeforeAfterSlider } from '../components/BeforeAfterSlider';
import { MetricCard } from '../components/MetricCard';
import { 
  ArrowRight, 
  Sliders, 
  CheckCircle2,
  ChevronRight
} from 'lucide-react';

interface HomeProps {
  onNavigate: (page: PageId) => void;
}

export const Home: React.FC<HomeProps> = ({ onNavigate }) => {
  return (
    <div className="space-y-24 py-8 sm:py-12">
      
      {/* ── HERO SECTION ──────────────────────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          
          {/* Left Text Column */}
          <div className="lg:col-span-6 space-y-6 text-left">
            
            {/* Research Tag */}
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-[#181F1B] border border-[#26322C] text-xs font-mono text-[#9CA8A0]">
              <span className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse"></span>
              <span className="text-[#34D399]">Research Showcase</span>
              <span className="text-[#6E7C73]">|</span>
              <span>Cityscapes Benchmark</span>
            </div>

            {/* Main Headline */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-[#F3F5F4] leading-[1.1]">
              Reimagining Streets <br className="hidden sm:inline" />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#10B981] via-[#34D399] to-emerald-200">
                for Everyone.
              </span>
            </h1>

            {/* Subtitle */}
            <p className="text-lg sm:text-xl text-[#9CA8A0] font-normal leading-relaxed max-w-xl">
              Human-in-the-Loop Generative AI for Inclusive Street Design.
              Transforming standard urban imagery into physically compliant, pedestrian-first streetscapes with deterministic accessibility validation.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <button
                onClick={() => onNavigate('demo')}
                className="flex items-center space-x-2 px-5 py-3 rounded-lg bg-[#10B981] hover:bg-[#059669] text-[#101412] font-semibold text-sm transition-all shadow-subtle-emerald hover:shadow-lg"
              >
                <span>Analyse a Street</span>
                <ArrowRight className="w-4 h-4" />
              </button>

              <button
                onClick={() => onNavigate('metrics')}
                className="flex items-center space-x-2 px-5 py-3 rounded-lg bg-[#181F1B] hover:bg-[#1D2521] text-[#F3F5F4] border border-[#26322C] hover:border-[#34423A] font-medium text-sm transition-all"
              >
                <Sliders className="w-4 h-4 text-[#34D399]" />
                <span>Technical Benchmarks</span>
              </button>
            </div>

            {/* Trust Badges / Verification summary */}
            <div className="pt-4 border-t border-[#26322C] grid grid-cols-3 gap-4 text-left font-mono">
              <div>
                <div className="text-xl font-bold text-[#F3F5F4]">0.1704</div>
                <div className="text-[11px] text-[#9CA8A0]">LPIPS Perceptual</div>
              </div>
              <div>
                <div className="text-xl font-bold text-[#34D399]">95%</div>
                <div className="text-[11px] text-[#9CA8A0]">ADA Compliance</div>
              </div>
              <div>
                <div className="text-xl font-bold text-[#F3F5F4]">3 / 3</div>
                <div className="text-[11px] text-[#9CA8A0]">Interventions/Scene</div>
              </div>
            </div>

          </div>

          {/* Right Column: Hero Visual Split Slider */}
          <div className="lg:col-span-6">
            <div className="relative rounded-xl p-2 bg-[#181F1B] border border-[#26322C] shadow-2xl">
              <BeforeAfterSlider
                beforeSrc="./samples/sample_before.png"
                afterSrc="./samples/sample_after.png"
                beforeLabel="Input: Multi-Lane Roadway"
                afterLabel="Synthesized: Continental Crosswalk"
                interventionType="crosswalk"
              />
              <div className="mt-2 px-2 py-1 flex items-center justify-between text-[11px] font-mono text-[#6E7C73]">
                <span>Cityscapes Validation Sample (1024×512)</span>
                <span className="text-[#34D399]">AccessNet Synthesized</span>
              </div>
            </div>
          </div>

        </div>
      </section>


      {/* ── FIVE-STAGE METHODOLOGY PREVIEW ──────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-12 space-y-3">
          <span className="text-xs font-mono uppercase tracking-wider text-[#10B981] bg-[#1D2521] px-3 py-1 rounded border border-[#10B981]/20">
            Pipeline Architecture
          </span>
          <h2 className="text-3xl font-bold text-[#F3F5F4] tracking-tight">
            5-Stage Inclusive Synthesis Pipeline
          </h2>
          <p className="text-sm text-[#9CA8A0] leading-relaxed">
            Unlike unconstrained diffusion models that hallucinate impossible infrastructure, StreetSynth couples deterministic computer vision heuristics with conditional GAN generation.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          
          {/* Stage 1 */}
          <div className="p-5 rounded-lg bg-[#181F1B] border border-[#26322C] hover:border-[#34423A] transition-all space-y-3">
            <div className="w-8 h-8 rounded bg-[#101412] border border-[#26322C] flex items-center justify-center font-mono text-xs font-bold text-[#34D399]">
              01
            </div>
            <h3 className="text-sm font-semibold text-[#F3F5F4]">Street Photo Input</h3>
            <p className="text-xs text-[#9CA8A0] leading-relaxed">
              Accepts standard monocular RGB street photography normalized to 1024×512 resolution.
            </p>
            <div className="text-[10px] font-mono text-[#6E7C73] pt-2 border-t border-[#26322C]">
              RGB Canvas
            </div>
          </div>

          {/* Stage 2 */}
          <div className="p-5 rounded-lg bg-[#181F1B] border border-[#26322C] hover:border-[#34423A] transition-all space-y-3">
            <div className="w-8 h-8 rounded bg-[#101412] border border-[#26322C] flex items-center justify-center font-mono text-xs font-bold text-[#34D399]">
              02
            </div>
            <h3 className="text-sm font-semibold text-[#F3F5F4]">Cityscapes Segmentation</h3>
            <p className="text-xs text-[#9CA8A0] leading-relaxed">
              SegFormer-B5 / DeepLabV3+ parses 19 discrete semantic classes (road, sidewalk, obstacles).
            </p>
            <div className="text-[10px] font-mono text-[#6E7C73] pt-2 border-t border-[#26322C]">
              mIoU: 0.667
            </div>
          </div>

          {/* Stage 3 */}
          <div className="p-5 rounded-lg bg-[#181F1B] border border-[#26322C] hover:border-[#34423A] transition-all space-y-3">
            <div className="w-8 h-8 rounded bg-[#101412] border border-[#26322C] flex items-center justify-center font-mono text-xs font-bold text-[#34D399]">
              03
            </div>
            <h3 className="text-sm font-semibold text-[#F3F5F4]">Monocular Depth</h3>
            <p className="text-xs text-[#9CA8A0] leading-relaxed">
              MiDaS / DPT-Large extracts dense relative depth fields for 3D ground plane slope validation.
            </p>
            <div className="text-[10px] font-mono text-[#6E7C73] pt-2 border-t border-[#26322C]">
              Continuous Depth [0, 1]
            </div>
          </div>

          {/* Stage 4 */}
          <div className="p-5 rounded-lg bg-[#1D2521] border border-[#10B981]/40 shadow-subtle-emerald transition-all space-y-3">
            <div className="w-8 h-8 rounded bg-[#10B981] text-[#101412] flex items-center justify-center font-mono text-xs font-bold">
              04
            </div>
            <h3 className="text-sm font-semibold text-[#34D399]">Placement Engine</h3>
            <p className="text-xs text-[#9CA8A0] leading-relaxed">
              Calculates boundary intersections, minimum span (&gt;15%), obstacle buffers, and curb alignments.
            </p>
            <div className="text-[10px] font-mono text-[#34D399] pt-2 border-t border-[#26322C]">
              Key Research Novelty
            </div>
          </div>

          {/* Stage 5 */}
          <div className="p-5 rounded-lg bg-[#181F1B] border border-[#26322C] hover:border-[#34423A] transition-all space-y-3">
            <div className="w-8 h-8 rounded bg-[#101412] border border-[#26322C] flex items-center justify-center font-mono text-xs font-bold text-[#34D399]">
              05
            </div>
            <h3 className="text-sm font-semibold text-[#F3F5F4]">AccessNet & LaMa</h3>
            <p className="text-xs text-[#9CA8A0] leading-relaxed">
              5-channel conditional GAN + LaMa cleanly inpaint and synthesize photorealistic interventions.
            </p>
            <div className="text-[10px] font-mono text-[#6E7C73] pt-2 border-t border-[#26322C]">
              LPIPS: 0.1704
            </div>
          </div>

        </div>

        <div className="mt-6 text-center">
          <button
            onClick={() => onNavigate('methodology')}
            className="inline-flex items-center space-x-1.5 text-xs font-mono text-[#34D399] hover:underline"
          >
            <span>Explore mathematical formulations and loss functions</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </section>


      {/* ── PLACEMENT ENGINE DEEP DIVE SECTION ──────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-gradient-to-br from-[#181F1B] to-[#141A17] border border-[#26322C] rounded-xl p-8 lg:p-12">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            
            <div className="lg:col-span-7 space-y-4 text-left">
              <span className="text-xs font-mono uppercase tracking-wider text-[#10B981] bg-[#1D2521] px-2.5 py-0.5 rounded border border-[#10B981]/30">
                Core Algorithmic Contribution
              </span>
              <h2 className="text-2xl sm:text-3xl font-bold text-[#F3F5F4] tracking-tight">
                The Accessibility Placement Engine
              </h2>
              <p className="text-sm text-[#9CA8A0] leading-relaxed">
                Rather than treating generative AI as an unguided black box, StreetSynth formulates accessibility as a constrained spatial optimization problem. The placement engine inspects multi-modal segmentation and depth masks to enforce physical safety standards before generation begins.
              </p>

              <div className="space-y-3 pt-2">
                <div className="flex items-start space-x-3">
                  <CheckCircle2 className="w-4 h-4 text-[#10B981] mt-0.5 shrink-0" />
                  <div>
                    <div className="text-xs font-mono font-semibold text-[#F3F5F4]">
                      Crosswalk Feasibility: Span &gt; 0.15 × W &amp; Flatness Variance σ &lt; 0.05
                    </div>
                    <div className="text-xs text-[#9CA8A0]">
                      Prevents placing zebra crossings over vertical curbs, barricades, or discontinuous roadway sections.
                    </div>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <CheckCircle2 className="w-4 h-4 text-[#10B981] mt-0.5 shrink-0" />
                  <div>
                    <div className="text-xs font-mono font-semibold text-[#F3F5F4]">
                      Sidewalk Seating: Obstacle Exclusion &amp; Maximum 20% Width Footprint
                    </div>
                    <div className="text-xs text-[#9CA8A0]">
                      Guarantees a continuous 1.8-meter ADA wheelchair passing lane by penalizing encroachment on active walkways.
                    </div>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <CheckCircle2 className="w-4 h-4 text-[#10B981] mt-0.5 shrink-0" />
                  <div>
                    <div className="text-xs font-mono font-semibold text-[#F3F5F4]">
                      Curb Ramp Alignment: 80th-Percentile Boundary Localization
                    </div>
                    <div className="text-xs text-[#9CA8A0]">
                      Identifies the exact road-sidewalk transition line to synthesize zero-step level boarding ramps.
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="lg:col-span-5">
              <div className="bg-[#101412] border border-[#26322C] rounded-lg p-5 font-mono text-xs space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-[#26322C]">
                  <span className="text-[#9CA8A0]">Placement Triage</span>
                  <span className="text-[#34D399] font-semibold">100% Satisfied</span>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-[#9CA8A0]">1. Connectivity (Crosswalk):</span>
                    <span className="text-[#34D399]">PASSED (1/1)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#9CA8A0]">2. Comfort (Bench Amenity):</span>
                    <span className="text-[#34D399]">PASSED (1/1)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#9CA8A0]">3. Mobility (Curb Ramp):</span>
                    <span className="text-[#34D399]">PASSED (1/1)</span>
                  </div>
                </div>

                <div className="pt-3 border-t border-[#26322C] text-[11px] text-[#6E7C73]">
                  Rule-based score: <strong className="text-[#F3F5F4]">3/3 Valid Interventions</strong>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>


      {/* ── PERFORMANCE RESULTS SUMMARY ────────────────────────────────────── */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between mb-8 gap-4">
          <div>
            <span className="text-xs font-mono uppercase tracking-wider text-[#10B981] bg-[#1D2521] px-2.5 py-0.5 rounded border border-[#10B981]/20">
              Empirical Evaluation
            </span>
            <h2 className="text-3xl font-bold text-[#F3F5F4] tracking-tight mt-2">
              Verified Technical Benchmarks
            </h2>
          </div>
          <button
            onClick={() => onNavigate('metrics')}
            className="flex items-center space-x-1 text-xs font-mono text-[#34D399] hover:underline"
          >
            <span>View full metrics dashboard</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            label="LPIPS Distance"
            value="0.1704"
            unit="score"
            subtitle="Low perceptual distortion against validation ground truth."
            improvement="+32.1% vs Pix2Pix"
            benchmark="0.2500"
            highlight={true}
          />
          <MetricCard
            label="Mean IoU (19 Classes)"
            value="0.667"
            unit="mIoU"
            subtitle="Accurate segmentation separating road, curb, and sidewalk."
            improvement="+2.7% semantic fidelity"
            benchmark="0.6400"
          />
          <MetricCard
            label="Road IoU"
            value="0.768"
            unit="IoU"
            subtitle="High-fidelity drivable corridor identification."
            benchmark="0.7200"
          />
          <MetricCard
            label="Constraint Compliance"
            value="95%"
            unit="compliance"
            subtitle="Interventions satisfying municipal accessibility standards."
            improvement="+27% vs unconstrained"
            benchmark="68%"
            highlight={true}
          />
        </div>
      </section>

    </div>
  );
};
