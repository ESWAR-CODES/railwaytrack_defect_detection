"""High-Level Railway Defect Inference and Inspection Orchestrator.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 3: End-to-End Visual Inspection & Inference Service.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from backend.engine.config_loader import ClassesConfig, load_classes_config
from backend.engine.severity_engine import (
    DefectDetection,
    FrameAnalysis,
    SeverityEngine,
    get_severity_engine,
)
from backend.engine.vision_annotator import (
    VisionAnnotator,
    get_vision_annotator,
)

logger = logging.getLogger("inference_service")


class InferenceService:
    """Orchestrates YOLO model prediction, deterministic severity evaluation, and visualization overlays."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[ClassesConfig] = None,
        fallback_model: str = "yolo11n.pt",
    ):
        self.config = config or load_classes_config()
        self.severity_engine = SeverityEngine(self.config)
        self.annotator = VisionAnnotator()
        self.fallback_model = fallback_model

        # Determine target model path
        default_model = Path(__file__).resolve().parent.parent / "models" / "best.pt"
        self.model_path = Path(model_path) if model_path else default_model

        self.model: Optional[Any] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model checkpoint or fallback to base model if custom weights do not exist."""
        if YOLO is None:
            logger.warning("Ultralytics library not available. InferenceService operating in mock mode.")
            return

        target_file = self.model_path
        # Check if custom best.pt exists and is a valid binary weight file (>10KB)
        if target_file.exists() and target_file.stat().st_size > 10240:
            try:
                logger.info(f"Loading custom trained YOLO model from: {target_file}")
                self.model = YOLO(str(target_file))
                return
            except Exception as e:
                logger.warning(f"Failed to load checkpoint at {target_file}: {e}")

        # Fallback to base pretrained model
        try:
            logger.info(f"Custom weights not available or mock file. Falling back to: {self.fallback_model}")
            self.model = YOLO(self.fallback_model)
        except Exception as e:
            logger.error(f"Failed to load fallback YOLO model: {e}")
            self.model = None

    def decode_image(self, source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """Decode input into a standard OpenCV BGR numpy array."""
        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            return source

        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"Input image path does not exist: {p}")
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"OpenCV failed to decode image file: {p}")
            return img

        if isinstance(source, (bytes, bytearray)):
            nparr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("OpenCV failed to decode image from byte buffer.")
            return img

        raise TypeError(f"Unsupported image input type: {type(source)}")

    def inspect(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        draw_hud: bool = True,
    ) -> Tuple[FrameAnalysis, np.ndarray, bytes]:
        """Execute end-to-end inspection: detection, severity analysis, overlay rendering, and JPEG encoding.
        
        Args:
            source: Image path, raw bytes, or numpy array.
            conf_threshold: Confidence filtering cutoff.
            iou_threshold: NMS overlap cutoff.
            draw_hud: Whether to render the telemetry HUD banner.
            
        Returns:
            Tuple of (FrameAnalysis, annotated_image_np, annotated_jpeg_bytes).
        """
        image = self.decode_image(source)
        img_h, img_w = image.shape[:2]

        raw_detections: List[Dict[str, Any]] = []

        if self.model is not None:
            try:
                results = self.model.predict(
                    source=image,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )

                if results and len(results) > 0:
                    first_res = results[0]
                    boxes = first_res.boxes
                    if boxes is not None and len(boxes) > 0:
                        cls_ids = boxes.cls.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        xyxy_arr = boxes.xyxy.cpu().numpy()
                        xywhn_arr = boxes.xywhn.cpu().numpy()
                        names_dict = getattr(first_res, "names", {})

                        for cid, conf_val, b_xyxy, b_norm in zip(
                            cls_ids, confs, xyxy_arr, xywhn_arr
                        ):
                            raw_name = names_dict.get(int(cid), str(int(cid)))
                            raw_detections.append(
                                {
                                    "class_id": int(cid),
                                    "class_name": raw_name,
                                    "confidence": float(conf_val),
                                    "bbox_xyxy": [float(c) for c in b_xyxy],
                                    "bbox_norm": [float(c) for c in b_norm],
                                }
                            )
            except Exception as e:
                logger.warning(f"YOLO predict invocation failed: {e}")

        # Evaluate severity and Track Risk Index
        analysis = self.severity_engine.evaluate_detections(raw_detections, image_shape=(img_h, img_w))

        # Render high-contrast visual annotations
        annotated_img = self.annotator.annotate_frame(image, analysis, draw_hud=draw_hud)

        # Encode to JPEG bytes
        success, encoded = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_bytes = encoded.tobytes() if success else b""

        return analysis, annotated_img, annotated_bytes

    def extract_crops(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        analysis: FrameAnalysis,
        padding: int = 12,
    ) -> List[Tuple[DefectDetection, np.ndarray]]:
        """Extract cropped thumbnail patches for each detected defect."""
        image = self.decode_image(source)
        return self.annotator.extract_defect_crops(image, analysis, padding=padding)
