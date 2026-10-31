import React from 'react';
import { PageId } from '../types';
import { Github, ExternalLink, ShieldCheck, HeartHandshake, Building2 } from 'lucide-react';

interface FooterProps {
  onNavigate: (page: PageId) => void;
}

export const Footer: React.FC<FooterProps> = ({ onNavigate }) => {
  return (
    <footer className="bg-[#0B0F0D] border-t border-[#26322C] text-[#9CA8A0] pt-12 pb-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Top Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 pb-10 border-b border-[#26322C]">
          
          {/* Col 1: Project Identity */}
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 rounded bg-[#10B981] flex items-center justify-center font-mono font-bold text-black text-xs">
                S
              </div>
              <span className="font-semibold text-[#F3F5F4] tracking-tight">StreetSynth</span>
              <span className="text-[11px] font-mono text-[#10B981] bg-[#1D2521] px-2 py-0.5 rounded border border-[#10B981]/20">
                Academic Research
              </span>
            </div>
            <p className="text-sm text-[#9CA8A0] max-w-md leading-relaxed">
              Human-in-the-Loop Generative AI for Inclusive Street Design. Combining monocular depth estimation,
              Cityscapes semantic segmentation, rule-based geometric feasibility validation, and 5-channel conditional GAN synthesis.
            </p>
            <div className="pt-2 flex items-center space-x-4 text-xs font-mono">
              <span className="text-[#F3F5F4]">Author: <span className="text-[#34D399]">Akhil Chivukula</span></span>
              <span className="text-[#6E7C73]">|</span>
              <span>PRN: 23070126009</span>
              <span className="text-[#6E7C73]">|</span>
              <span>AIML A1</span>
            </div>
          </div>

          {/* Col 2: Navigation Links */}
          <div className="space-y-3">
            <h4 className="text-xs font-mono uppercase tracking-wider text-[#F3F5F4] font-semibold">
              Research Sections
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <button onClick={() => onNavigate('home')} className="hover:text-[#34D399] transition-colors">
                  Overview & Hero
                </button>
              </li>
              <li>
                <button onClick={() => onNavigate('demo')} className="hover:text-[#34D399] transition-colors">
                  Interactive Workspace
                </button>
              </li>
              <li>
                <button onClick={() => onNavigate('metrics')} className="hover:text-[#34D399] transition-colors">
                  Benchmarks & Evaluation
                </button>
              </li>
              <li>
                <button onClick={() => onNavigate('methodology')} className="hover:text-[#34D399] transition-colors">
                  5-Stage Architecture
                </button>
              </li>
              <li>
                <button onClick={() => onNavigate('about')} className="hover:text-[#34D399] transition-colors">
                  About & SDG Targets
                </button>
              </li>
            </ul>
          </div>

          {/* Col 3: SDG & Open Source */}
          <div className="space-y-3">
            <h4 className="text-xs font-mono uppercase tracking-wider text-[#F3F5F4] font-semibold">
              Civic Alignment
            </h4>
            <div className="space-y-2 text-xs">
              <div className="flex items-start space-x-2 p-2 rounded bg-[#181F1B] border border-[#26322C]">
                <HeartHandshake className="w-4 h-4 text-[#10B981] mt-0.5 shrink-0" />
                <div>
                  <div className="font-semibold text-[#F3F5F4]">SDG 10: Reduced Inequalities</div>
                  <div className="text-[#9CA8A0]">Barrier-free mobility for disabled and elderly citizens.</div>
                </div>
              </div>
              <div className="flex items-start space-x-2 p-2 rounded bg-[#181F1B] border border-[#26322C]">
                <Building2 className="w-4 h-4 text-[#34D399] mt-0.5 shrink-0" />
                <div>
                  <div className="font-semibold text-[#F3F5F4]">SDG 11: Sustainable Cities</div>
                  <div className="text-[#9CA8A0]">Safe, walk-first public space retrofits.</div>
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* Bottom Bar */}
        <div className="pt-6 flex flex-col sm:flex-row items-center justify-between text-xs text-[#6E7C73] space-y-3 sm:space-y-0">
          <div className="flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-[#10B981]" />
            <span>Open Source Academic Research | MIT License</span>
          </div>

          <div className="flex items-center space-x-4">
            <a 
              href="https://github.com/sirchivvi/StreetSynth" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center space-x-1 text-[#9CA8A0] hover:text-[#34D399] transition-colors"
            >
              <Github className="w-3.5 h-3.5" />
              <span>sirchivvi/StreetSynth</span>
              <ExternalLink className="w-2.5 h-2.5 ml-0.5" />
            </a>
            <a 
              href="https://huggingface.co/spaces/SirChivvi/StreetSynth" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-[#9CA8A0] hover:text-[#34D399] transition-colors"
            >
              HF Space
            </a>
          </div>
        </div>

      </div>
    </footer>
  );
};

