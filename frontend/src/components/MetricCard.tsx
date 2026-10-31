import React from 'react';

interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  subtitle?: string;
  improvement?: string;
  benchmark?: string;
  icon?: React.ReactNode;
  highlight?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  subtitle,
  improvement,
  benchmark,
  icon,
  highlight = false,
}) => {
  return (
    <div 
      className={`p-5 rounded-lg border transition-all ${
        highlight 
          ? 'bg-[#1D2521] border-[#10B981]/40 shadow-subtle-emerald' 
          : 'bg-[#181F1B] border-[#26322C] hover:border-[#34423A]'
      }`}
    >
      <div className="flex items-center justify-between text-xs text-[#9CA8A0] font-mono mb-2">
        <span className="flex items-center space-x-1.5">
          {icon && <span className="text-[#34D399]">{icon}</span>}
          <span>{label}</span>
        </span>
        {unit && (
          <span className="px-1.5 py-0.5 rounded bg-[#1D2521] text-[#9CA8A0] border border-[#26322C] text-[10px]">
            {unit}
          </span>
        )}
      </div>

      <div className="flex items-baseline space-x-2 my-1">
        <span className="text-3xl font-bold font-mono tracking-tight text-[#F3F5F4]">
          {value}
        </span>
        {improvement && (
          <span className="text-xs font-mono font-medium text-[#34D399] bg-[#10B981]/10 px-1.5 py-0.5 rounded border border-[#10B981]/20">
            {improvement}
          </span>
        )}
      </div>

      {subtitle && (
        <p className="text-xs text-[#9CA8A0] mt-2 leading-relaxed">
          {subtitle}
        </p>
      )}

      {benchmark && (
        <div className="mt-3 pt-2 border-t border-[#26322C] flex items-center justify-between text-[11px] font-mono text-[#6E7C73]">
          <span>Baseline:</span>
          <span className="text-[#9CA8A0]">{benchmark}</span>
        </div>
      )}
    </div>
  );
};

