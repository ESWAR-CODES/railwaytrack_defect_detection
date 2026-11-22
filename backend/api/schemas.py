"""Pydantic Request and Response Schemas for Railway Defect Inspection API.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 4: FastAPI Inference & Inspection Backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """API health status and model availability."""
    status: str = Field(default="healthy", description="Overall service operational status")
    model_loaded: bool = Field(..., description="Whether YOLO model weights are loaded")
    weights_path: Optional[str] = Field(None, description="Path to active model weights file")
    model_name: str = Field(..., description="Active model identifier or filename")
    classes: Dict[int, str] = Field(..., description="Canonical defect classes {id: name}")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of health query")


class DetectionItem(BaseModel):
    """Single defect bounding box detection and severity evaluation."""
    class_id: int = Field(..., description="Canonical integer class identifier (0-4)")
    class_name: str = Field(..., description="Canonical defect class name")
    confidence: float = Field(..., description="Model confidence score between 0.0 and 1.0")
    bbox_xyxy: List[float] = Field(..., description="Pixel coordinates [x1, y1, x2, y2]")
    bbox_norm: List[float] = Field(..., description="Normalized coordinates [x_center, y_center, width, height]")
    area_ratio: float = Field(..., description="Normalized box area width * height")
    diagonal_ratio: float = Field(..., description="Normalized box diagonal length")
    severity: str = Field(..., description="Evaluated severity: LOW, MEDIUM, or HIGH")
    color_hex: str = Field(..., description="Severity color code in hex format")
    recommended_action: str = Field(..., description="Maintenance guidance for this defect")


class InspectionSummary(BaseModel):
    """Aggregated frame-level inspection defect summary and Track Risk Index."""
    total_defects: int = Field(..., description="Total number of defects detected in frame")
    counts_by_class: Dict[str, int] = Field(default_factory=dict, description="Defect counts broken down by class")
    counts_by_severity: Dict[str, int] = Field(default_factory=dict, description="Defect counts broken down by severity")
    track_risk_index: float = Field(..., description="Composite Track Risk Index clamped to [0.0, 100.0]")
    overall_risk_level: str = Field(..., description="Overall risk category: NONE, LOW, MEDIUM, or HIGH")
    overall_color_hex: str = Field(default="#10B981", description="Telemetry color corresponding to overall risk")
    recommended_action: str = Field(default="", description="Recommended action for the overall track state")


class InspectionResponse(BaseModel):
    """Complete response payload for a processed railway image inspection."""
    inspection_id: str = Field(..., description="Unique UUID for this inspection event")
    filename: str = Field(..., description="Original filename of the uploaded image")
    image_width: int = Field(..., description="Original image width in pixels")
    image_height: int = Field(..., description="Original image height in pixels")
    confidence_threshold: float = Field(..., description="Confidence threshold used during inference")
    summary: InspectionSummary = Field(..., description="Composite severity and risk summary")
    detections: List[DetectionItem] = Field(default_factory=list, description="List of detected defect items")
    annotated_image_base64: Optional[str] = Field(None, description="Base64-encoded annotated JPEG image with HUD strip")
    crops_base64: List[str] = Field(default_factory=list, description="Base64-encoded JPEG thumbnail crops of each defect")
    created_at: str = Field(..., description="ISO 8601 UTC timestamp of inspection creation")


class MetricsResponse(BaseModel):
    """Evaluation metrics report for trained railway defect YOLO checkpoint."""
    status: str = Field(..., description="Evaluation status: trained_and_evaluated, available, or unavailable")
    timestamp: Optional[str] = Field(None, description="Timestamp when evaluation was executed")
    model_path: Optional[str] = Field(None, description="Model checkpoint file evaluated")
    device: Optional[str] = Field(None, description="Execution device used for evaluation (cpu / cuda)")
    training_parameters: Optional[Dict[str, Any]] = Field(None, description="Hyperparameters used during training")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Detailed mAP, precision, and recall metrics")
    message: Optional[str] = Field(None, description="Informative status message if metrics are unavailable")


class InspectionHistoryItem(BaseModel):
    """Summary item representing a historical inspection."""
    inspection_id: str = Field(..., description="Unique inspection identifier")
    filename: str = Field(..., description="Original image filename")
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")
    total_defects: int = Field(..., description="Total defect count")
    track_risk_index: float = Field(..., description="Computed Track Risk Index (0-100)")
    overall_risk_level: str = Field(..., description="Overall risk level (NONE, LOW, MEDIUM, HIGH)")
    created_at: str = Field(..., description="ISO 8601 UTC timestamp")


class InspectionHistoryResponse(BaseModel):
    """Response containing a list of recent inspections."""
    total_inspections: int = Field(..., description="Total count of historical inspections in store")
    items: List[InspectionHistoryItem] = Field(default_factory=list, description="List of inspection history items")


class SampleItem(BaseModel):
    """Pre-loaded sample dataset image for live testing."""
    id: str = Field(..., description="Unique sample identifier")
    filename: str = Field(..., description="Image filename on disk or virtual synthetic file")
    split: str = Field(..., description="Dataset split (test, val, train, synthetic)")
    url: str = Field(..., description="Direct URL endpoint to fetch raw image bytes")
    label: Optional[str] = Field(None, description="Human-readable defect hint or label summary")


class SampleListResponse(BaseModel):
    """List of available railway track samples for immediate demo execution."""
    samples: List[SampleItem] = Field(default_factory=list, description="Available railway sample images")

