import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Columns, SplitSquareVertical, Download } from 'lucide-react';

interface BeforeAfterSliderProps {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
  aspectRatio?: string;
  className?: string;
  interventionType?: string;
}

export const BeforeAfterSlider: React.FC<BeforeAfterSliderProps> = ({
  beforeSrc,
  afterSrc,
  beforeLabel = "Original Street",
  afterLabel = "AccessNet Intervention",
  aspectRatio = "aspect-[2/1]",
  className = "",
  interventionType
}) => {
  const [sliderPosition, setSliderPosition] = useState(50); // percentage 0 - 100
  const [isDragging, setIsDragging] = useState(false);
  const [viewMode, setViewMode] = useState<'slider' | 'side-by-side'>('slider');
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(percentage);
  }, []);

  const handleTouchMove = useCallback((e: TouchEvent) => {
    if (!isDragging) return;
    handleMove(e.touches[0].clientX);
  }, [isDragging, handleMove]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    handleMove(e.clientX);
  }, [isDragging, handleMove]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp, handleTouchMove]);

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = afterSrc;
    link.download = `streetsynth-${interventionType || 'generated'}.png`;
    link.click();
  };

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Controls Bar */}
      <div className="flex items-center justify-between text-xs text-[#9CA8A0] font-mono px-1">
        <div className="flex items-center space-x-2">
          <span className="inline-block w-2 h-2 rounded-full bg-[#10B981]"></span>
          <span>Interactive Comparison</span>
          {interventionType && (
            <span className="px-2 py-0.5 rounded bg-[#1D2521] text-[#34D399] border border-[#26322C] capitalize">
              {interventionType.replace('_', ' ')}
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {/* Mode Switcher */}
          <div className="bg-[#181F1B] p-0.5 rounded border border-[#26322C] flex items-center">
            <button
              onClick={() => setViewMode('slider')}
              className={`px-2 py-1 rounded flex items-center space-x-1 transition-colors ${
                viewMode === 'slider' ? 'bg-[#1D2521] text-[#34D399]' : 'hover:text-[#F3F5F4]'
              }`}
              title="Split comparison slider"
            >
              <SplitSquareVertical className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Slider</span>
            </button>
            <button
              onClick={() => setViewMode('side-by-side')}
              className={`px-2 py-1 rounded flex items-center space-x-1 transition-colors ${
                viewMode === 'side-by-side' ? 'bg-[#1D2521] text-[#34D399]' : 'hover:text-[#F3F5F4]'
              }`}
              title="Side-by-side comparison"
            >
              <Columns className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Side by Side</span>
            </button>
          </div>

          {/* Download Button */}
          <button
            onClick={handleDownload}
            className="p-1.5 rounded bg-[#181F1B] hover:bg-[#1D2521] text-[#9CA8A0] hover:text-[#34D399] border border-[#26322C] transition-colors"
            title="Download generated image"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Display Container */}
      {viewMode === 'slider' ? (
        <div
          ref={containerRef}
          onMouseDown={() => setIsDragging(true)}
          onTouchStart={() => setIsDragging(true)}
          className={`relative w-full ${aspectRatio} rounded-lg overflow-hidden border border-[#26322C] bg-[#141A17] select-none cursor-ew-resize group shadow-card`}
        >
          {/* Background Image: Generated (After) */}
          <img
            src={afterSrc}
            alt={afterLabel}
            className="absolute inset-0 w-full h-full object-cover pointer-events-none"
            loading="lazy"
          />

          {/* Foreground Image: Original (Before) clipped to sliderPosition */}
          <div
            className="absolute inset-0 overflow-hidden pointer-events-none"
            style={{ width: `${sliderPosition}%` }}
          >
            <img
              src={beforeSrc}
              alt={beforeLabel}
              className="absolute inset-0 w-full h-full object-cover pointer-events-none"
              style={{
                width: containerRef.current ? `${containerRef.current.clientWidth}px` : '100%',
                maxWidth: 'none',
              }}
              loading="lazy"
            />
          </div>

          {/* Vertical Divider Line */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-[#34D399] shadow-[0_0_10px_#10B981] pointer-events-none transition-opacity"
            style={{ left: `${sliderPosition}%` }}
          >
            {/* Center Handle Knob */}
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-[#181F1B] border-2 border-[#10B981] flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
              <div className="flex space-x-1">
                <div className="w-0.5 h-3 bg-[#34D399]"></div>
                <div className="w-0.5 h-3 bg-[#34D399]"></div>
              </div>
            </div>
          </div>

          {/* Top Floating Badges */}
          <div className="absolute top-3 left-3 pointer-events-none">
            <span className="px-2.5 py-1 rounded bg-[#101412]/80 backdrop-blur-sm text-[11px] font-mono font-medium text-[#9CA8A0] border border-[#26322C]">
              {beforeLabel}
            </span>
          </div>

          <div className="absolute top-3 right-3 pointer-events-none">
            <span className="px-2.5 py-1 rounded bg-[#101412]/80 backdrop-blur-sm text-[11px] font-mono font-semibold text-[#34D399] border border-[#10B981]/30">
              {afterLabel}
            </span>
          </div>

          {/* Drag instruction overlay for mobile/hover */}
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
            <span className="px-3 py-1 rounded-full bg-[#101412]/90 backdrop-blur-sm text-[10px] font-mono text-[#9CA8A0] border border-[#26322C]">
              ← Drag slider to compare →
            </span>
          </div>
        </div>
      ) : (
        /* Side by Side Mode */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="relative rounded-lg overflow-hidden border border-[#26322C] bg-[#141A17]">
            <img src={beforeSrc} alt={beforeLabel} className="w-full h-auto object-cover" />
            <div className="absolute top-3 left-3">
              <span className="px-2.5 py-1 rounded bg-[#101412]/80 backdrop-blur-sm text-[11px] font-mono text-[#9CA8A0] border border-[#26322C]">
                {beforeLabel}
              </span>
            </div>
          </div>

          <div className="relative rounded-lg overflow-hidden border border-[#26322C] bg-[#141A17]">
            <img src={afterSrc} alt={afterLabel} className="w-full h-auto object-cover" />
            <div className="absolute top-3 right-3">
              <span className="px-2.5 py-1 rounded bg-[#101412]/80 backdrop-blur-sm text-[11px] font-mono text-[#34D399] border border-[#10B981]/30">
                {afterLabel}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
