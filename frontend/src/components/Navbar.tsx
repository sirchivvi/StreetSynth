import React, { useState } from 'react';
import { PageId, BackendStatus } from '../types';
import { 
  Layers, 
  BarChart2, 
  Cpu, 
  Info, 
  Github, 
  ExternalLink, 
  Menu, 
  X, 
  Sparkles,
  Server
} from 'lucide-react';

interface NavbarProps {
  currentPage: PageId;
  onNavigate: (page: PageId) => void;
  backendStatus: BackendStatus;
}

export const Navbar: React.FC<NavbarProps> = ({ currentPage, onNavigate, backendStatus }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems: { id: PageId; label: string; icon: React.ReactNode }[] = [
    { id: 'home', label: 'Overview', icon: <Sparkles className="w-4 h-4" /> },
    { id: 'demo', label: 'Analysis & Demo', icon: <Layers className="w-4 h-4" /> },
    { id: 'metrics', label: 'Research Metrics', icon: <BarChart2 className="w-4 h-4" /> },
    { id: 'methodology', label: 'Methodology', icon: <Cpu className="w-4 h-4" /> },
    { id: 'about', label: 'About & SDGs', icon: <Info className="w-4 h-4" /> },
  ];

  const handleNav = (id: PageId) => {
    onNavigate(id);
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <header className="sticky top-0 z-50 bg-[#101412]/90 backdrop-blur-md border-b border-[#26322C] transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Research Badge */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => handleNav('home')}>
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#10B981] to-[#064E3B] flex items-center justify-center shadow-subtle-emerald border border-[#34D399]/30">
              <span className="font-mono font-bold text-white text-base">S</span>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-semibold text-lg text-[#F3F5F4] tracking-tight hover:text-[#34D399] transition-colors">
                  StreetSynth
                </span>
                <span className="px-1.5 py-0.5 text-[10px] uppercase font-mono tracking-wider font-semibold bg-[#1D2521] text-[#10B981] border border-[#10B981]/30 rounded">
                  Research
                </span>
              </div>
            </div>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1 lg:space-x-2">
            {navItems.map((item) => {
              const isActive = currentPage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleNav(item.id)}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-[#1D2521] text-[#34D399] border border-[#26322C] shadow-sm'
                      : 'text-[#9CA8A0] hover:text-[#F3F5F4] hover:bg-[#181F1B]'
                  }`}
                >
                  {item.icon}
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Right Action: Status Pill & Links */}
          <div className="hidden sm:flex items-center space-x-3">
            {/* Backend Connectivity Status Pill */}
            <div 
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-mono border ${
                backendStatus.online 
                  ? 'bg-[#10B981]/10 text-[#34D399] border-[#10B981]/30' 
                  : 'bg-[#1D2521] text-[#9CA8A0] border-[#26322C]'
              }`}
              title={backendStatus.online ? `Inference ready on ${backendStatus.device?.toUpperCase() || 'CPU'}` : 'Showcase Mode: Full interactive sample datasets active'}
            >
              <div className={`w-2 h-2 rounded-full ${backendStatus.online ? 'bg-[#10B981] animate-pulse' : 'bg-amber-500/70'}`} />
              <Server className="w-3 h-3 text-[#9CA8A0]" />
              <span className="hidden lg:inline">
                {backendStatus.online ? `Backend: ${backendStatus.device?.toUpperCase() || 'ONLINE'}` : 'Showcase Mode'}
              </span>
            </div>

            {/* Hugging Face Link */}
            <a
              href="https://huggingface.co/spaces/SirChivvi/StreetSynth"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-[#9CA8A0] hover:text-[#34D399] hover:bg-[#181F1B] rounded-md transition-colors"
              title="View on Hugging Face Spaces"
            >
              <span className="text-sm font-mono font-medium">HF</span>
            </a>

            {/* GitHub Link */}
            <a
              href="https://github.com/sirchivvi/StreetSynth"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-1.5 px-3 py-1.5 bg-[#181F1B] hover:bg-[#1D2521] text-[#F3F5F4] border border-[#26322C] hover:border-[#34423A] rounded-md text-xs font-mono transition-all"
            >
              <Github className="w-3.5 h-3.5 text-[#34D399]" />
              <span>GitHub</span>
              <ExternalLink className="w-2.5 h-2.5 text-[#9CA8A0]" />
            </a>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex md:hidden items-center space-x-2">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 text-[#9CA8A0] hover:text-[#F3F5F4] bg-[#181F1B] rounded-md border border-[#26322C]"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>

        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[#181F1B] border-b border-[#26322C] px-4 pt-2 pb-4 space-y-1">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => handleNav(item.id)}
              className={`w-full flex items-center space-x-2.5 px-3 py-2 rounded-md text-sm font-medium ${
                currentPage === item.id
                  ? 'bg-[#1D2521] text-[#34D399] border border-[#26322C]'
                  : 'text-[#9CA8A0] hover:text-[#F3F5F4] hover:bg-[#101412]'
              }`}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
          <div className="pt-3 border-t border-[#26322C] flex items-center justify-between text-xs text-[#9CA8A0]">
            <span className="font-mono">
              Status: {backendStatus.online ? 'Backend Connected' : 'Sample Cache Active'}
            </span>
            <a 
              href="https://github.com/sirchivvi/StreetSynth" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="text-[#34D399] flex items-center space-x-1"
            >
              <Github className="w-3.5 h-3.5" />
              <span>sirchivvi</span>
            </a>
          </div>
        </div>
      )}
    </header>
  );
};

