/**
 * Strict TypeScript Type Definitions mirroring Backend API Schemas.
 * 
 * Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
 * Phase 5: React Frontend & Full-Stack Integration.
 */

export type SeverityLevel = "LOW" | "MEDIUM" | "HIGH";
export type OverallRiskCategory = "NONE" | "LOW" | "MEDIUM" | "HIGH";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  weights_path: string | null;
  model_name: string;
  classes: Record<string, string>;
  timestamp: string;
}

export interface DetectionItem {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
  bbox_norm: [number, number, number, number];
  area_ratio: number;
  diagonal_ratio: number;
  severity: SeverityLevel;
  color_hex: string;
  recommended_action: string;
}

export interface InspectionSummary {
  total_defects: number;
  counts_by_class: Record<string, number>;
  counts_by_severity: Record<string, number>;
  track_risk_index: number;
  overall_risk_level: OverallRiskCategory;
  overall_color_hex: string;
  recommended_action: string;
}

export interface InspectionResponse {
  inspection_id: string;
  filename: string;
  image_width: number;
  image_height: number;
  confidence_threshold: number;
  summary: InspectionSummary;
  detections: DetectionItem[];
  annotated_image_base64: string | null;
  crops_base64: string[];
  created_at: string;
}

export interface MetricsResponse {
  status: string;
  timestamp?: string | null;
  model_path?: string | null;
  device?: string | null;
  training_parameters?: {
    model_name?: string;
    dataset_yaml?: string;
    epochs?: number;
    batch?: number;
    imgsz?: number;
    device?: string;
    fraction?: number;
    seed?: number;
    [key: string]: any;
  } | null;
  metrics?: {
    mAP50?: number;
    mAP50_95?: number;
    precision?: number;
    recall?: number;
    per_class_map?: Record<string, number>;
    [key: string]: any;
  } | null;
  message?: string | null;
}

export interface InspectionHistoryItem {
  inspection_id: string;
  filename: string;
  image_width: number;
  image_height: number;
  total_defects: number;
  track_risk_index: number;
  overall_risk_level: string;
  created_at: string;
}

export interface InspectionHistoryResponse {
  total_inspections: number;
  items: InspectionHistoryItem[];
}
