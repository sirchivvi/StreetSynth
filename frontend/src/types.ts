export type PageId = 'home' | 'demo' | 'metrics' | 'methodology' | 'about';

export type InterventionType = 'crosswalk' | 'bench' | 'curb_ramp';

export interface IndicatorItem {
  score: number;
  max: number;
  reason: string;
  passed?: boolean;
}

export interface Indicators {
  connectivity: IndicatorItem;
  comfort: IndicatorItem;
  mobility: IndicatorItem;
  overall: {
    score: number;
    max: number;
    percentage?: number;
  };
}

export interface PlacementData {
  valid: boolean;
  x: number;
  y: number;
  w: number;
  h: number;
  span?: number;
  flatness?: number;
  reason: string;
}

export interface SampleItem {
  id: string;
  title: string;
  intervention: InterventionType;
  location: string;
  description: string;
  before_image: string;
  after_image: string;
  segmentation_image?: string;
  depth_image?: string;
  placement_overlay?: string;
  placement: PlacementData;
  indicators: Indicators;
  metrics: {
    lpips: number;
    segmentation_iou: number;
    constraint_satisfaction: string;
  };
}

export interface BackendStatus {
  online: boolean;
  device?: string;
  cuda_available?: boolean;
  checkpoint_available?: boolean;
  pipeline_loaded?: boolean;
  models?: Record<string, string>;
  latency_ms?: number;
}

