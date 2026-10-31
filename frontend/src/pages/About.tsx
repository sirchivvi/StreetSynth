import React from 'react';
import { 
  CheckCircle2, 
  AlertTriangle, 
  Github, 
  ExternalLink,
  BookOpen
} from 'lucide-react';

export const About: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-16">
      
      {/* Header */}
      <div className="space-y-3 pb-6 border-b border-[#26322C]">
        <div className="flex items-center space-x-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#10B981]"></span>
          <span className="text-xs font-mono uppercase tracking-wider text-[#10B981]">
            Academic Research &amp; Civic Impact
          </span>
        </div>
        <h1 className="text-3xl font-bold text-[#F3F5F4] tracking-tight">
          About StreetSynth
        </h1>
        <p className="text-sm text-[#9CA8A0] max-w-3xl leading-relaxed">
          Pioneering human-in-the-loop generative artificial intelligence for inclusive, barrier-free urban street redesign.
        </p>
      </div>


      {/* ── RESEARCHER ATTRIBUTION CARD ─────────────────────────────────────── */}
      <div className="bg-gradient-to-r from-[#181F1B] to-[#141A17] border border-[#26322C] rounded-xl p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6">
          <div className="flex items-center space-x-4">
            <div className="w-14 h-14 rounded-full bg-gradient-to-br from-[#10B981] to-[#064E3B] border-2 border-[#34D399]/40 flex items-center justify-center font-mono font-bold text-xl text-white shadow-subtle-emerald">
              AC
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="text-xl font-bold text-[#F3F5F4]">Akhil Chivukula</h3>
                <span className="px-2 py-0.5 rounded bg-[#10B981]/15 text-[#34D399] border border-[#10B981]/30 text-xs font-mono">
                  Primary Researcher
                </span>
              </div>
              <p className="text-xs font-mono text-[#9CA8A0] mt-1">
                PRN: 23070126009 | Division: AIML A1
              </p>
              <p className="text-xs text-[#6E7C73] mt-1">
                Artificial Intelligence &amp; Machine Learning Specialization
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <a
              href="https://github.com/sirchivvi/StreetSynth"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-[#1D2521] hover:bg-[#26322C] text-[#F3F5F4] border border-[#26322C] text-xs font-mono transition-all"
            >
              <Github className="w-4 h-4 text-[#34D399]" />
              <span>sirchivvi/StreetSynth</span>
              <ExternalLink className="w-3 h-3 text-[#9CA8A0]" />
            </a>

            <a
              href="https://huggingface.co/spaces/SirChivvi/StreetSynth"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-[#10B981] hover:bg-[#059669] text-[#101412] text-xs font-mono font-semibold transition-all"
            >
              <span>HF Space</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </div>


      {/* ── MOTIVATION & PROBLEM STATEMENT ──────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Research Motivation */}
        <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-6 space-y-4">
          <div className="flex items-center space-x-2 text-[#34D399]">
            <BookOpen className="w-4 h-4" />
            <h3 className="text-sm font-semibold font-mono uppercase tracking-wider">
              Research Motivation
            </h3>
          </div>
          <p className="text-sm text-[#9CA8A0] leading-relaxed">
            Over 1.3 billion individuals worldwide experience significant disability. Modern urban streetscapes remain overwhelmingly engineered around vehicular throughput, creating insurmountable physical barriers for wheelchair users, visually impaired pedestrians, and elderly citizens.
          </p>
          <p className="text-sm text-[#9CA8A0] leading-relaxed">
            Traditional municipal redesign requires months of on-site surveyor auditing, manual CAD drafting, and bureaucratic public reviews. Visualizing the tangible impact of accessibility retrofits in photorealistic quality accelerates municipal decision-making and community advocacy.
          </p>
        </div>

        {/* Problem Statement */}
        <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-6 space-y-4">
          <div className="flex items-center space-x-2 text-[#34D399]">
            <AlertTriangle className="w-4 h-4" />
            <h3 className="text-sm font-semibold font-mono uppercase tracking-wider">
              Technical Problem Statement
            </h3>
          </div>
          <p className="text-sm text-[#9CA8A0] leading-relaxed">
            While generic diffusion models and unconditioned GANs can generate artistic street visuals, they lack physical geometry understanding: they frequently float crosswalks across curbs, block pedestrian passageways with oversized seating, and ignore wheelchair ramp slope standards.
          </p>
          <p className="text-sm text-[#9CA8A0] leading-relaxed">
            StreetSynth bridges this critical divide by coupling monocular depth estimation and semantic segmentation with deterministic spatial rules—ensuring every synthesized intervention is structurally compliant and physically feasible.
          </p>
        </div>

      </div>


      {/* ── SUSTAINABLE DEVELOPMENT GOALS (SDG 10 & 11) ─────────────────────── */}
      <div className="space-y-6">
        <div>
          <span className="text-xs font-mono uppercase tracking-wider text-[#10B981]">
            Global Policy Alignment
          </span>
          <h2 className="text-2xl font-bold text-[#F3F5F4] tracking-tight mt-1">
            United Nations Sustainable Development Goals
          </h2>
          <p className="text-sm text-[#9CA8A0] mt-1 leading-relaxed">
            StreetSynth directly supports the United Nations 2030 Agenda for Sustainable Development.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* SDG 10 */}
          <div className="p-6 rounded-lg bg-[#181F1B] border border-[#26322C] space-y-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-lg bg-[#E5243B]/20 border border-[#E5243B]/40 flex items-center justify-center font-bold text-[#E5243B] font-mono text-base">
                10
              </div>
              <div>
                <h3 className="text-base font-bold text-[#F3F5F4]">
                  SDG 10: Reduced Inequalities
                </h3>
                <span className="text-xs font-mono text-[#9CA8A0]">
                  Target 10.2: Universal Social &amp; Physical Inclusion
                </span>
              </div>
            </div>

            <p className="text-xs text-[#9CA8A0] leading-relaxed">
              Empowering persons with reduced mobility, elderly pedestrians, and parents with strollers by automating the detection and remediation of physical urban infrastructure inequalities.
            </p>

            <ul className="text-xs font-mono text-[#9CA8A0] space-y-1.5 pt-2 border-t border-[#26322C]">
              <li className="flex items-center space-x-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#10B981]" />
                <span>Zero-step sidewalk curb transitions</span>
              </li>
              <li className="flex items-center space-x-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#10B981]" />
                <span>ADA compliant pedestrian crossing paths</span>
              </li>
            </ul>
          </div>

          {/* SDG 11 */}
          <div className="p-6 rounded-lg bg-[#181F1B] border border-[#26322C] space-y-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-lg bg-[#FD9D24]/20 border border-[#FD9D24]/40 flex items-center justify-center font-bold text-[#FD9D24] font-mono text-base">
                11
              </div>
              <div>
                <h3 className="text-base font-bold text-[#F3F5F4]">
                  SDG 11: Sustainable Cities &amp; Communities
                </h3>
                <span className="text-xs font-mono text-[#9CA8A0]">
                  Target 11.2 &amp; 11.7: Safe, Accessible Green &amp; Public Spaces
                </span>
              </div>
            </div>

            <p className="text-xs text-[#9CA8A0] leading-relaxed">
              Transitioning automotive-centric streets into pedestrianized, human-scaled spaces with accessible resting furniture, active transit connectivity, and universal street safety.
            </p>

            <ul className="text-xs font-mono text-[#9CA8A0] space-y-1.5 pt-2 border-t border-[#26322C]">
              <li className="flex items-center space-x-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#34D399]" />
                <span>Walkable urban neighborhood infrastructure</span>
              </li>
              <li className="flex items-center space-x-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-[#34D399]" />
                <span>Rapid participatory civic consensus tooling</span>
              </li>
            </ul>
          </div>

        </div>
      </div>


      {/* ── TECHNICAL CONTRIBUTIONS SUMMARY ─────────────────────────────────── */}
      <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-6 sm:p-8 space-y-4">
        <h3 className="text-lg font-bold text-[#F3F5F4]">
          Key Technical Novelties
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono text-[#9CA8A0]">
          <div className="p-4 rounded bg-[#101412] border border-[#26322C] space-y-2">
            <div className="text-[#34D399] font-bold text-sm">5-Channel Tensor Conditioning</div>
            <p className="text-[11px] leading-relaxed">
              Concatenating raw RGB pixels, SegFormer semantic class distributions, and continuous DPT depth into an unified tensor input for perspective-accurate synthesis.
            </p>
          </div>

          <div className="p-4 rounded bg-[#101412] border border-[#26322C] space-y-2">
            <div className="text-[#34D399] font-bold text-sm">Deterministic Placement Engine</div>
            <p className="text-[11px] leading-relaxed">
              Rule-based spatial constraint validation enforcing road span (&gt;15%), sidewalk width clear corridors (1.8m), and curb boundary alignment before inpainting.
            </p>
          </div>

          <div className="p-4 rounded bg-[#101412] border border-[#26322C] space-y-2">
            <div className="text-[#34D399] font-bold text-sm">Depth Consistency Loss</div>
            <p className="text-[11px] leading-relaxed">
              Edge-aware gradient penalty tying synthetic marking gradients to physical ground-plane variations, preventing floating or warped interventions.
            </p>
          </div>
        </div>
      </div>


      {/* ── FUTURE SCOPE ────────────────────────────────────────────────────── */}
      <div className="bg-[#141A17] border border-[#26322C] rounded-lg p-6 sm:p-8 space-y-4">
        <h3 className="text-lg font-bold text-[#F3F5F4]">
          Future Research Directions
        </h3>
        <p className="text-xs text-[#9CA8A0] leading-relaxed">
          The StreetSynth roadmap includes three planned expansions:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono text-[#9CA8A0]">
          <div className="p-3 rounded bg-[#101412] border border-[#26322C]">
            <strong className="text-[#F3F5F4]">1. 360° Panoramic Panoramas:</strong> Multi-view panoramic stitching to redesign entire urban intersections concurrently.
          </div>
          <div className="p-3 rounded bg-[#101412] border border-[#26322C]">
            <strong className="text-[#F3F5F4]">2. Municipal GIS Integration:</strong> Direct vector export to ArcGIS and OpenStreetMap for urban planning department workflows.
          </div>
          <div className="p-3 rounded bg-[#101412] border border-[#26322C]">
            <strong className="text-[#F3F5F4]">3. 3D Gaussian Splatting:</strong> Converting synthesized 2D interventions into real-time interactive 3D virtual walkthroughs.
          </div>
        </div>
      </div>

    </div>
  );
};
