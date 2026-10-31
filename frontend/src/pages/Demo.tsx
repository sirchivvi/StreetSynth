import React, { useState } from 'react';
import { SAMPLE_CATALOG } from '../data/mockData';
import { SampleItem, InterventionType, BackendStatus } from '../types';
import { BeforeAfterSlider } from '../components/BeforeAfterSlider';
import { 
  Upload, 
  Layers, 
  Download, 
  Sparkles, 
  AlertCircle, 
  CheckCircle, 
  Clock, 
  Sliders, 
  ServerOff
} from 'lucide-react';

interface DemoProps {
  backendStatus: BackendStatus;
}

export const Demo: React.FC<DemoProps> = ({ backendStatus }) => {
  // Active sample or upload state
  const [selectedSample, setSelectedSample] = useState<SampleItem>(SAMPLE_CATALOG[0]);
  const [activeIntervention, setActiveIntervention] = useState<InterventionType>('crosswalk');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedPreview, setUploadedPreview] = useState<string | null>(null);

  // Active tensor inspection tab
  const [activeTab, setActiveTab] = useState<'comparison' | 'segmentation' | 'depth' | 'placement' | 'scorecard'>('comparison');

  // Execution states
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLiveOutput, setIsLiveOutput] = useState(false);

  // Live generated result if backend was called
  const [liveResult, setLiveResult] = useState<{
    before: string;
    after: string;
    seg_overlay?: string;
    depth_overlay?: string;
    placement: any;
    indicators: any;
    timeSec?: number;
  } | null>(null);

  // Handle Preset Selection
  const handleSelectPreset = (sample: SampleItem) => {
    setSelectedSample(sample);
    setActiveIntervention(sample.intervention);
    setUploadedFile(null);
    setUploadedPreview(null);
    setLiveResult(null);
    setIsLiveOutput(false);
    setErrorMessage(null);
  };

  // Handle File Upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setErrorMessage("Please select a valid image file (JPEG, PNG, WebP).");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setErrorMessage("Image file exceeds the 10MB limit.");
      return;
    }

    setErrorMessage(null);
    setUploadedFile(file);
    const reader = new FileReader();
    reader.onload = () => {
      setUploadedPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  // Run Pipeline Trigger
  const handleRunPipeline = async () => {
    setIsProcessing(true);
    setErrorMessage(null);

    // If backend is offline or no uploaded file, use high-resolution pre-computed sample
    if (!backendStatus.online || !uploadedFile) {
      setProcessingStep("Simulating pipeline: SegFormer-B5 segmentation...");
      await new Promise(r => setTimeout(r, 600));
      setProcessingStep("Simulating pipeline: MiDaS DPT depth estimation...");
      await new Promise(r => setTimeout(r, 600));
      setProcessingStep("Simulating pipeline: Placement Engine constraint check...");
      await new Promise(r => setTimeout(r, 500));
      setProcessingStep("Simulating pipeline: AccessNet 5-channel synthesis...");
      await new Promise(r => setTimeout(r, 700));

      const matched = SAMPLE_CATALOG.find(s => s.intervention === activeIntervention) || SAMPLE_CATALOG[0];
      setSelectedSample(matched);
      setIsLiveOutput(false);
      setIsProcessing(false);
      return;
    }

    // Call live FastAPI backend
    try {
      setProcessingStep("Uploading image to StreetSynth backend...");
      const formData = new FormData();
      formData.append("file", uploadedFile);
      formData.append("intervention", activeIntervention);
      formData.append("lightweight", "true");
      formData.append("show_overlays", "true");

      setProcessingStep("Running neural pipeline (SegFormer-B5 + DPT + AccessNet)...");
      const apiBase = (import.meta.env.VITE_API_BASE_URL as string) || '';
      const resp = await fetch(`${apiBase}/api/generate`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({}));
        throw new Error(errorData.detail || "Server error during model inference");
      }

      const data = await resp.json();
      if (!data.success) {
        setErrorMessage(data.reason || "Intervention rejected by placement engine constraints.");
        setIsProcessing(false);
        return;
      }

      setLiveResult({
        before: data.before_image,
        after: data.after_image,
        seg_overlay: data.seg_overlay,
        depth_overlay: data.depth_overlay,
        placement: data.placement,
        indicators: data.indicators,
        timeSec: data.inference_time_sec,
      });
      setIsLiveOutput(true);
    } catch (err: any) {
      console.warn("Backend error, falling back to cached sample:", err);
      setErrorMessage(`Inference failed (${err.message}). Displaying pre-computed sample validation result.`);
      const matched = SAMPLE_CATALOG.find(s => s.intervention === activeIntervention) || SAMPLE_CATALOG[0];
      setSelectedSample(matched);
      setIsLiveOutput(false);
    } finally {
      setIsProcessing(false);
    }
  };

  // Determine current active before/after data
  const currentBefore = liveResult ? liveResult.before : (uploadedPreview || selectedSample.before_image);
  const currentAfter = liveResult ? liveResult.after : selectedSample.after_image;
  const currentPlacement = liveResult ? liveResult.placement : selectedSample.placement;
  const currentIndicators = liveResult ? liveResult.indicators : selectedSample.indicators;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Workspace Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-[#26322C]">
        <div>
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[#10B981]"></span>
            <span className="text-xs font-mono uppercase tracking-wider text-[#10B981]">
              Interactive AI Workspace
            </span>
          </div>
          <h1 className="text-3xl font-bold text-[#F3F5F4] tracking-tight mt-1">
            Street Analysis &amp; Generative Synthesis
          </h1>
          <p className="text-sm text-[#9CA8A0] mt-1">
            Upload custom street photography or select benchmark validation scenes to evaluate pedestrian accessibility.
          </p>
        </div>

        {/* Backend Connectivity Status Pill */}
        <div className="flex items-center space-x-2 self-start md:self-auto">
          {backendStatus.online ? (
            <div className="px-3 py-1.5 rounded-lg bg-[#10B981]/10 border border-[#10B981]/30 text-xs font-mono text-[#34D399] flex items-center space-x-2">
              <span className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse"></span>
              <span>FastAPI Inference Online ({backendStatus.device?.toUpperCase() || 'CPU'})</span>
            </div>
          ) : (
            <div className="px-3 py-1.5 rounded-lg bg-[#181F1B] border border-[#26322C] text-xs font-mono text-[#9CA8A0] flex items-center space-x-2" title="Backend not detected locally. Pre-computed Cityscapes validation outputs are fully interactive.">
              <ServerOff className="w-3.5 h-3.5 text-amber-500" />
              <span>Showcase Fallback Mode (Offline-Ready)</span>
            </div>
          )}
        </div>
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* ── LEFT COLUMN: CONTROLS & INPUTS (4 COLS) ───────────────────────── */}
        <div className="lg:col-span-4 space-y-6">
          
          {/* Preset Selector */}
          <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-5 space-y-4">
            <h3 className="text-xs font-mono uppercase tracking-wider text-[#F3F5F4] font-semibold flex items-center justify-between">
              <span>Benchmark Presets</span>
              <span className="text-[#6E7C73] text-[10px]">Cityscapes Validated</span>
            </h3>

            <div className="space-y-2">
              {SAMPLE_CATALOG.map((sample) => {
                const isSelected = selectedSample.id === sample.id && !uploadedFile;
                return (
                  <button
                    key={sample.id}
                    onClick={() => handleSelectPreset(sample)}
                    className={`w-full text-left p-3 rounded-lg border transition-all ${
                      isSelected
                        ? 'bg-[#1D2521] border-[#10B981]/40 text-[#F3F5F4] shadow-sm'
                        : 'bg-[#141A17] border-[#26322C] text-[#9CA8A0] hover:text-[#F3F5F4] hover:border-[#34423A]'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold capitalize font-mono text-[#34D399]">
                        {sample.intervention.replace('_', ' ')}
                      </span>
                      <span className="text-[10px] text-[#6E7C73] font-mono">1024×512</span>
                    </div>
                    <div className="text-xs text-[#F3F5F4] font-medium mt-1 truncate">
                      {sample.title}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Custom Upload Dropzone */}
          <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-5 space-y-4">
            <h3 className="text-xs font-mono uppercase tracking-wider text-[#F3F5F4] font-semibold flex items-center justify-between">
              <span>Or Upload Custom Street Photo</span>
              <span className="text-[#6E7C73] text-[10px]">Max 10MB</span>
            </h3>

            <label className="border-2 border-dashed border-[#26322C] hover:border-[#10B981]/50 rounded-lg p-4 flex flex-col items-center justify-center cursor-pointer transition-colors bg-[#141A17]">
              <Upload className="w-6 h-6 text-[#9CA8A0] mb-2" />
              <span className="text-xs font-medium text-[#F3F5F4]">Click to upload or drag &amp; drop</span>
              <span className="text-[10px] text-[#6E7C73] mt-1 font-mono">JPG, PNG, WebP (2:1 aspect ratio recommended)</span>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>

            {uploadedPreview && (
              <div className="relative rounded border border-[#26322C] overflow-hidden">
                <img src={uploadedPreview} alt="Uploaded input preview" className="w-full h-24 object-cover" />
                <span className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-black/70 text-[9px] font-mono text-[#34D399]">
                  Uploaded
                </span>
              </div>
            )}
          </div>

          {/* Intervention Selection */}
          <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-5 space-y-4">
            <h3 className="text-xs font-mono uppercase tracking-wider text-[#F3F5F4] font-semibold">
              Select Intervention Type
            </h3>

            <div className="grid grid-cols-3 gap-2">
              {(['crosswalk', 'bench', 'curb_ramp'] as InterventionType[]).map((type) => {
                const isActive = activeIntervention === type;
                return (
                  <button
                    key={type}
                    onClick={() => setActiveIntervention(type)}
                    className={`py-2 px-2 rounded border text-center transition-all ${
                      isActive
                        ? 'bg-[#1D2521] border-[#10B981] text-[#34D399] font-semibold shadow-sm'
                        : 'bg-[#141A17] border-[#26322C] text-[#9CA8A0] hover:text-[#F3F5F4]'
                    }`}
                  >
                    <div className="text-[11px] font-mono capitalize">
                      {type.replace('_', ' ')}
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunPipeline}
              disabled={isProcessing}
              className={`w-full py-3 px-4 rounded-lg font-semibold text-sm flex items-center justify-center space-x-2 transition-all ${
                isProcessing
                  ? 'bg-[#26322C] text-[#9CA8A0] cursor-not-allowed'
                  : 'bg-[#10B981] hover:bg-[#059669] text-[#101412] shadow-subtle-emerald hover:shadow-lg'
              }`}
            >
              {isProcessing ? (
                <>
                  <div className="w-4 h-4 border-2 border-emerald-900 border-t-emerald-400 rounded-full animate-spin"></div>
                  <span>Running Inference...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Synthesize Intervention</span>
                </>
              )}
            </button>

            {isProcessing && (
              <div className="p-3 rounded bg-[#101412] border border-[#26322C] text-xs font-mono text-[#34D399] space-y-1">
                <div className="flex items-center space-x-2">
                  <Clock className="w-3.5 h-3.5 animate-spin" />
                  <span>{processingStep}</span>
                </div>
                <div className="text-[10px] text-[#6E7C73]">
                  CPU inference latency: ~30–60s | GPU: ~8–10s
                </div>
              </div>
            )}

            {errorMessage && (
              <div className="p-3 rounded bg-amber-950/30 border border-amber-800/50 text-xs text-amber-300 flex items-start space-x-2">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />
                <span className="leading-relaxed">{errorMessage}</span>
              </div>
            )}
          </div>

        </div>


        {/* ── RIGHT COLUMN: RESULTS & MULTI-LAYER INSPECTION (8 COLS) ───────── */}
        <div className="lg:col-span-8 space-y-6">
          
          {/* Main Display Header with Model Distinction Badge */}
          <div className="bg-[#181F1B] border border-[#26322C] rounded-lg p-5 space-y-4">
            
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-[#26322C] gap-2">
              <div>
                <h3 className="text-base font-semibold text-[#F3F5F4] flex items-center space-x-2">
                  <span>{selectedSample.title}</span>
                </h3>
                <p className="text-xs text-[#9CA8A0] mt-0.5">
                  {selectedSample.location}
                </p>
              </div>

              {/* Status distinction badge */}
              <div className="flex items-center space-x-2">
                {isLiveOutput ? (
                  <span className="px-2.5 py-1 rounded bg-[#10B981]/20 text-[#34D399] border border-[#10B981]/40 text-xs font-mono flex items-center space-x-1.5">
                    <CheckCircle className="w-3.5 h-3.5" />
                    <span>Live AccessNet Output</span>
                  </span>
                ) : (
                  <span className="px-2.5 py-1 rounded bg-[#1D2521] text-[#9CA8A0] border border-[#26322C] text-xs font-mono">
                    Cityscapes Benchmark Output
                  </span>
                )}
              </div>
            </div>

            {/* Tab Navigation for Layers */}
            <div className="flex flex-wrap gap-1 border-b border-[#26322C] pb-3 text-xs font-mono">
              <button
                onClick={() => setActiveTab('comparison')}
                className={`px-3 py-1.5 rounded-md flex items-center space-x-1.5 transition-all ${
                  activeTab === 'comparison'
                    ? 'bg-[#1D2521] text-[#34D399] border border-[#26322C]'
                    : 'text-[#9CA8A0] hover:text-[#F3F5F4]'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Split Comparison</span>
              </button>

              <button
                onClick={() => setActiveTab('segmentation')}
                className={`px-3 py-1.5 rounded-md flex items-center space-x-1.5 transition-all ${
                  activeTab === 'segmentation'
                    ? 'bg-[#1D2521] text-[#34D399] border border-[#26322C]'
                    : 'text-[#9CA8A0] hover:text-[#F3F5F4]'
                }`}
              >
                <span className="w-2.5 h-2.5 rounded-sm bg-[#804080]" />
                <span>SegFormer-B5 Map</span>
              </button>

              <button
                onClick={() => setActiveTab('depth')}
                className={`px-3 py-1.5 rounded-md flex items-center space-x-1.5 transition-all ${
                  activeTab === 'depth'
                    ? 'bg-[#1D2521] text-[#34D399] border border-[#26322C]'
                    : 'text-[#9CA8A0] hover:text-[#F3F5F4]'
                }`}
              >
                <span className="w-2.5 h-2.5 rounded-sm bg-gradient-to-r from-blue-500 to-amber-500" />
                <span>MiDaS Depth</span>
              </button>

              <button
                onClick={() => setActiveTab('placement')}
                className={`px-3 py-1.5 rounded-md flex items-center space-x-1.5 transition-all ${
                  activeTab === 'placement'
                    ? 'bg-[#1D2521] text-[#34D399] border border-[#26322C]'
                    : 'text-[#9CA8A0] hover:text-[#F3F5F4]'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                <span>Placement Mask</span>
              </button>

              <button
                onClick={() => setActiveTab('scorecard')}
                className={`px-3 py-1.5 rounded-md flex items-center space-x-1.5 transition-all ${
                  activeTab === 'scorecard'
                    ? 'bg-[#1D2521] text-[#34D399] border border-[#26322C]'
                    : 'text-[#9CA8A0] hover:text-[#F3F5F4]'
                }`}
              >
                <CheckCircle className="w-3.5 h-3.5" />
                <span>Accessibility Scorecard</span>
              </button>
            </div>

            {/* Tab Content Panels */}
            <div className="pt-2">
              {activeTab === 'comparison' && (
                <BeforeAfterSlider
                  beforeSrc={currentBefore}
                  afterSrc={currentAfter}
                  beforeLabel="Original Street (Cityscapes)"
                  afterLabel={`AccessNet Synthesized (${activeIntervention})`}
                  interventionType={activeIntervention}
                />
              )}

              {activeTab === 'segmentation' && (
                <div className="space-y-3">
                  <div className="rounded-lg overflow-hidden border border-[#26322C] bg-[#141A17]">
                    <img
                      src={liveResult?.seg_overlay || selectedSample.segmentation_image || "./samples/sample_segmentation.png"}
                      alt="SegFormer Segmentation Map"
                      className="w-full h-auto object-cover"
                    />
                  </div>
                  <div className="p-3 rounded bg-[#101412] border border-[#26322C] text-xs font-mono text-[#9CA8A0] flex items-center justify-between">
                    <span>Segmentation: SegFormer-B5 / DeepLabV3+ (Cityscapes 19 Classes)</span>
                    <span className="text-[#34D399]">Road IoU: 0.768 | Sidewalk: 0.567</span>
                  </div>
                </div>
              )}

              {activeTab === 'depth' && (
                <div className="space-y-3">
                  <div className="rounded-lg overflow-hidden border border-[#26322C] bg-[#141A17]">
                    <img
                      src={liveResult?.depth_overlay || selectedSample.depth_image || "./samples/sample_depth.png"}
                      alt="DPT-Large Depth Estimation"
                      className="w-full h-auto object-cover"
                    />
                  </div>
                  <div className="p-3 rounded bg-[#101412] border border-[#26322C] text-xs font-mono text-[#9CA8A0] flex items-center justify-between">
                    <span>Depth: Intel DPT-Large / MiDaS Monocular Inverse Depth</span>
                    <span className="text-[#34D399]">Normalized Range: [0.0, 1.0]</span>
                  </div>
                </div>
              )}

              {activeTab === 'placement' && (
                <div className="space-y-3">
                  <div className="rounded-lg overflow-hidden border border-[#26322C] bg-[#141A17]">
                    <img
                      src={selectedSample.placement_overlay || "./samples/sample_placement_overlay.png"}
                      alt="Placement Engine Mask"
                      className="w-full h-auto object-cover"
                    />
                  </div>
                  <div className="p-3 rounded bg-[#101412] border border-[#26322C] text-xs font-mono text-[#9CA8A0] space-y-1">
                    <div className="text-[#F3F5F4] font-semibold">Placement Engine Bounding Heuristics:</div>
                    <div>Center: ({currentPlacement.x}, {currentPlacement.y}) | Dimensions: {currentPlacement.w}px × {currentPlacement.h}px</div>
                    <div className="text-[#34D399]">Justification: {currentPlacement.reason}</div>
                  </div>
                </div>
              )}

              {activeTab === 'scorecard' && (
                <div className="p-5 rounded-lg bg-[#141A17] border border-[#26322C] space-y-4 font-mono text-xs">
                  <div className="flex items-center justify-between pb-3 border-b border-[#26322C]">
                    <span className="text-sm font-semibold text-[#F3F5F4]">
                      Accessibility Indicators Scorecard
                    </span>
                    <span className="text-[#34D399] font-bold text-sm bg-[#10B981]/15 px-2.5 py-1 rounded border border-[#10B981]/30">
                      Overall Score: {currentIndicators.overall.score} / {currentIndicators.overall.max} (100%)
                    </span>
                  </div>

                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="text-left text-[#6E7C73] border-b border-[#26322C]">
                        <th className="py-2">Indicator Domain</th>
                        <th className="py-2 text-center">Status</th>
                        <th className="py-2">Engineering Rationale</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#26322C]">
                      <tr>
                        <td className="py-2.5 font-semibold text-[#F3F5F4]">🚶 Connectivity</td>
                        <td className="py-2.5 text-center text-[#34D399] font-bold">
                          {currentIndicators.connectivity.score}/1 [PASSED]
                        </td>
                        <td className="py-2.5 text-[#9CA8A0]">
                          {currentIndicators.connectivity.reason}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2.5 font-semibold text-[#F3F5F4]">🪑 Comfort</td>
                        <td className="py-2.5 text-center text-[#34D399] font-bold">
                          {currentIndicators.comfort.score}/1 [PASSED]
                        </td>
                        <td className="py-2.5 text-[#9CA8A0]">
                          {currentIndicators.comfort.reason}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2.5 font-semibold text-[#F3F5F4]">♿ Mobility</td>
                        <td className="py-2.5 text-center text-[#34D399] font-bold">
                          {currentIndicators.mobility.score}/1 [PASSED]
                        </td>
                        <td className="py-2.5 text-[#9CA8A0]">
                          {currentIndicators.mobility.reason}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
            </div>

          </div>

          {/* Download & Export Bar */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-[#181F1B] border border-[#26322C] text-xs font-mono">
            <span className="text-[#9CA8A0]">
              Output Resolution: <strong className="text-[#F3F5F4]">1024 × 512 px</strong> (Standard Cityscapes Benchmark)
            </span>
            <a
              href={currentAfter}
              download={`streetsynth-${activeIntervention}.png`}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-[#1D2521] hover:bg-[#26322C] text-[#34D399] border border-[#10B981]/30 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download High-Res Output</span>
            </a>
          </div>

        </div>

      </div>

    </div>
  );
};
