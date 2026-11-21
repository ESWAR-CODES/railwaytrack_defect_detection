"""FastAPI Route Handlers for Railway Defect Inspection API.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 4: FastAPI Inference & Inspection Backend.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from backend.api.schemas import (
    DetectionItem,
    HealthResponse,
    InspectionHistoryItem,
    InspectionHistoryResponse,
    InspectionResponse,
    InspectionSummary,
    MetricsResponse,
)
from backend.api.store import InspectionStore, get_inspection_store
from backend.engine.config_loader import ClassesConfig, load_classes_config
from backend.engine.inference_service import InferenceService

logger = logging.getLogger("api_routes")

router = APIRouter(prefix="/api/v1", tags=["Inspection & Monitoring"])

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Global InferenceService singleton
_service_instance: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Access the global inference service instance (gracefully initializes)."""
    global _service_instance
    if _service_instance is None:
        try:
            _service_instance = InferenceService()
        except Exception as e:
            logger.error(f"Error initializing InferenceService: {e}")
            _service_instance = InferenceService(fallback_model="yolo11n.pt")
    return _service_instance


def set_inference_service(service: Optional[InferenceService]) -> None:
    """Set or reset the global inference service instance (useful for testing)."""
    global _service_instance
    _service_instance = service


@router.get("/health", response_model=HealthResponse)
def get_health(
    service: InferenceService = Depends(get_inference_service),
) -> HealthResponse:
    """Return operational health status, active weights availability, and canonical defect classes."""
    model_loaded = service.model is not None
    weights_path = str(service.model_path) if service.model_path.exists() else None
    model_name = (
        service.model_path.name
        if (service.model_path and service.model_path.exists())
        else service.fallback_model
    )
    canonical_classes = {
        int(k): str(v) for k, v in service.config.classes.items()
    }
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        weights_path=weights_path,
        model_name=model_name,
        classes=canonical_classes,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/inspect", response_model=InspectionResponse)
async def inspect_image(
    file: UploadFile = File(..., description="Railway image file (.jpg, .jpeg, .png, .webp)"),
    conf_threshold: Optional[float] = Query(default=None, description="Confidence detection cutoff (0.05 - 0.95)"),
    form_conf_threshold: Optional[float] = Form(default=None, alias="conf_threshold"),
    include_crops: Optional[bool] = Query(default=None, description="Whether to include thumbnail crops"),
    form_include_crops: Optional[bool] = Form(default=None, alias="include_crops"),
    draw_hud: Optional[bool] = Query(default=None, description="Whether to render telemetry HUD banner"),
    form_draw_hud: Optional[bool] = Form(default=None, alias="draw_hud"),
    service: InferenceService = Depends(get_inference_service),
    store: InspectionStore = Depends(get_inspection_store),
) -> InspectionResponse:
    """Run full computer vision defect inspection on an uploaded drone/track image.
    
    Returns structured detections, deterministic severity ratings, Track Risk Index,
    base64-encoded annotated visual overlay, and thumbnail patches for localized review.
    """
    # Resolve parameter precedence (Query param takes precedence over Form param, falling back to default)
    raw_conf = (
        conf_threshold
        if conf_threshold is not None
        else (form_conf_threshold if form_conf_threshold is not None else 0.25)
    )
    should_include_crops = (
        include_crops
        if include_crops is not None
        else (form_include_crops if form_include_crops is not None else True)
    )
    should_draw_hud = (
        draw_hud
        if draw_hud is not None
        else (form_draw_hud if form_draw_hud is not None else True)
    )

    # Validate and clamp confidence threshold between 0.05 and 0.95
    clamped_conf = max(0.05, min(0.95, float(raw_conf)))

    # Validate file extension
    filename = file.filename or "uploaded_image.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Read and decode image buffer
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not contents or len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    try:
        image_np = service.decode_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or unreadable image file: {e}")

    img_h, img_w = image_np.shape[:2]

    # Execute inspection pipeline
    try:
        analysis, annotated_np, annotated_bytes = service.inspect(
            source=image_np,
            conf_threshold=clamped_conf,
            draw_hud=should_draw_hud,
        )
    except Exception as e:
        logger.error(f"Inference execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {e}")

    # Encode annotated image to base64
    annotated_base64 = base64.b64encode(annotated_bytes).decode("utf-8") if annotated_bytes else None

    # Extract thumbnail crops if requested
    crops_base64: List[str] = []
    if should_include_crops and analysis.total_defects > 0:
        try:
            crop_tuples = service.extract_crops(image_np, analysis)
            for _, crop_np in crop_tuples:
                success, enc_crop = cv2.imencode(".jpg", crop_np, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if success:
                    crops_base64.append(base64.b64encode(enc_crop.tobytes()).decode("utf-8"))
        except Exception as e:
            logger.warning(f"Crop extraction warning: {e}")

    # Build Pydantic response models
    summary = InspectionSummary(
        total_defects=analysis.total_defects,
        counts_by_class=analysis.counts_by_class,
        counts_by_severity=analysis.counts_by_severity,
        track_risk_index=analysis.track_risk_index,
        overall_risk_level=analysis.overall_risk_level,
        overall_color_hex=analysis.overall_color_hex,
        recommended_action=analysis.recommended_action,
    )

    detections = [
        DetectionItem(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=round(d.confidence, 4),
            bbox_xyxy=[round(c, 2) for c in d.bbox_xyxy],
            bbox_norm=[round(c, 6) for c in d.bbox_norm],
            area_ratio=d.area_ratio,
            diagonal_ratio=d.diagonal_ratio,
            severity=d.severity,
            color_hex=d.color_hex,
            recommended_action=d.recommended_action,
        )
        for d in analysis.detections
    ]

    inspection_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    response = InspectionResponse(
        inspection_id=inspection_id,
        filename=filename,
        image_width=img_w,
        image_height=img_h,
        confidence_threshold=clamped_conf,
        summary=summary,
        detections=detections,
        annotated_image_base64=annotated_base64,
        crops_base64=crops_base64,
        created_at=created_at,
    )

    # Persist summary to history store
    history_record = InspectionHistoryItem(
        inspection_id=inspection_id,
        filename=filename,
        image_width=img_w,
        image_height=img_h,
        total_defects=analysis.total_defects,
        track_risk_index=analysis.track_risk_index,
        overall_risk_level=analysis.overall_risk_level,
        created_at=created_at,
    )
    store.add(history_record)

    return response


def get_metrics_path() -> Path:
    """Resolve path to the trained model evaluation metrics JSON report."""
    return Path(__file__).resolve().parent.parent / "models" / "evaluation_metrics.json"


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """Retrieve model training evaluation metrics (mAP50, mAP50-95, precision, recall, per-class mAP)."""
    metrics_path = get_metrics_path()

    if not metrics_path.exists():
        return MetricsResponse(
            status="unavailable",
            message="No evaluation metrics found. Model training has not been executed yet.",
        )

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return MetricsResponse(
            status=data.get("status", "available"),
            timestamp=data.get("timestamp"),
            model_path=data.get("model_path"),
            device=data.get("device"),
            training_parameters=data.get("training_parameters"),
            metrics=data.get("metrics"),
            message="Evaluation metrics loaded successfully.",
        )
    except Exception as e:
        logger.error(f"Failed to read evaluation metrics file: {e}")
        return MetricsResponse(
            status="error",
            message=f"Failed to parse evaluation metrics file: {e}",
        )


@router.get("/history", response_model=InspectionHistoryResponse)
def get_history(
    limit: int = Query(default=20, ge=1, le=100, description="Max historical inspection records to return"),
    store: InspectionStore = Depends(get_inspection_store),
) -> InspectionHistoryResponse:
    """Retrieve recent inspection events (newest first)."""
    items = store.list_recent(limit=limit)
    return InspectionHistoryResponse(
        total_inspections=store.count(),
        items=items,
    )
