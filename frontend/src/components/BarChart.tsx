import React from 'react';

interface BarItem {
  name: string;
  iou: number;
  color?: string;
}

interface BarChartProps {
  data: BarItem[];
  title?: string;
  subtitle?: string;
}

export const BarChart: React.FC<BarChartProps> = ({ data, title, subtitle }) => {
  return (
    <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-5">
      {title && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-[#F3F5F4]">{title}</h4>
          {subtitle && <p className="text-xs text-[#9CA8A0] mt-0.5">{subtitle}</p>}
        </div>
      )}

      <div className="space-y-3 font-mono text-xs">
        {data.map((item) => {
          const percentage = Math.round(item.iou * 100);
          const isKeyTarget = item.name === "Road" || item.name === "Sidewalk";
          return (
            <div key={item.name} className="space-y-1">
              <div className="flex justify-between items-center text-[11px]">
                <span className={`flex items-center space-x-1.5 ${isKeyTarget ? 'text-[#34D399] font-semibold' : 'text-[#9CA8A0]'}`}>
                  <span 
                    className="w-2.5 h-2.5 rounded-sm inline-block shrink-0" 
                    style={{ backgroundColor: item.color || '#10B981' }} 
                  />
                  <span>{item.name}</span>
                  {isKeyTarget && (
                    <span className="text-[9px] px-1 py-0.2 rounded bg-[#10B981]/20 text-[#34D399]">
                      Key Target
                    </span>
                  )}
                </span>
                <span className="text-[#F3F5F4]">{item.iou.toFixed(3)} ({percentage}%)</span>
              </div>

              {/* Bar track */}
              <div className="w-full h-2 rounded bg-[#101412] overflow-hidden border border-[#26322C]">
                <div
                  className={`h-full rounded-sm transition-all duration-500 ${
                    isKeyTarget ? 'bg-gradient-to-r from-[#10B981] to-[#34D399]' : 'bg-[#4D5F54]'
                  }`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

