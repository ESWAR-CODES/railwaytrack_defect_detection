"""Unit and Integration Tests for Defect Severity Engine, Vision Annotator, and YOLO Trainer.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 3: Test Suite.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.engine.config_loader import load_classes_config
from backend.engine.inference_service import InferenceService
from backend.engine.severity_engine import (
    RECOMMENDED_ACTIONS,
    FrameAnalysis,
    OverallRiskLevel,
    SeverityEngine,
)
from backend.engine.vision_annotator import VisionAnnotator, hex_to_bgr
from backend.engine.yolo_trainer import (
    YoloTrainConfig,
    detect_best_device,
    extract_evaluation_metrics,
    train_railway_model,
)


@pytest.fixture
def severity_engine() -> SeverityEngine:
    """Fixture providing freshly initialized SeverityEngine."""
    return SeverityEngine(load_classes_config())


@pytest.fixture
def vision_annotator() -> VisionAnnotator:
    """Fixture providing VisionAnnotator."""
    return VisionAnnotator()


class TestSeverityEngine:
    """Tests verifying geometric metric calculation and deterministic defect severity classification."""

    def test_geometry_computation(self, severity_engine: SeverityEngine):
        """Verify normalized width, height, area_ratio, and diagonal_ratio computation."""
        w, h, area, diag = severity_engine.compute_geometry([0.5, 0.5, 0.04, 0.03])
        assert w == 0.04
        assert h == 0.03
        assert area == pytest.approx(0.0012, rel=1e-4)
        assert diag == pytest.approx(0.05, rel=1e-4)

    def test_rail_crack_severity_thresholds(self, severity_engine: SeverityEngine):
        """rail_crack uses diagonal_ratio: LOW < 0.03, MEDIUM <= 0.08, HIGH > 0.08."""
        # LOW: diagonal = 0.02
        det_low = [{"class_name": "rail_crack", "bbox_norm": [0.5, 0.5, 0.012, 0.016]}]  # diag = 0.02
        res_low = severity_engine.evaluate_detections(det_low)
        assert res_low.detections[0].severity == "LOW"

        # MEDIUM: diagonal = 0.05
        det_med = [{"class_name": "rail_crack", "bbox_norm": [0.5, 0.5, 0.03, 0.04]}]  # diag = 0.05
        res_med = severity_engine.evaluate_detections(det_med)
        assert res_med.detections[0].severity == "MEDIUM"

        # HIGH: diagonal = 0.10
        det_high = [{"class_name": "rail_crack", "bbox_norm": [0.5, 0.5, 0.06, 0.08]}]  # diag = 0.10
        res_high = severity_engine.evaluate_detections(det_high)
        assert res_high.detections[0].severity == "HIGH"

    def test_rail_corrosion_severity_thresholds(self, severity_engine: SeverityEngine):
        """rail_corrosion uses area_ratio: LOW < 0.05, MEDIUM <= 0.15, HIGH > 0.15."""
        # LOW: area = 0.02
        det_low = [{"class_name": "rail_corrosion", "bbox_norm": [0.5, 0.5, 0.1, 0.2]}]
        res_low = severity_engine.evaluate_detections(det_low)
        assert res_low.detections[0].severity == "LOW"

        # MEDIUM: area = 0.10
        det_med = [{"class_name": "rail_corrosion", "bbox_norm": [0.5, 0.5, 0.2, 0.5]}]
        res_med = severity_engine.evaluate_detections(det_med)
        assert res_med.detections[0].severity == "MEDIUM"

        # HIGH: area = 0.25
        det_high = [{"class_name": "rail_corrosion", "bbox_norm": [0.5, 0.5, 0.5, 0.5]}]
        res_high = severity_engine.evaluate_detections(det_high)
        assert res_high.detections[0].severity == "HIGH"

    def test_missing_fastener_count_escalation(self, severity_engine: SeverityEngine):
        """missing_fastener escalates with frame count: 1==LOW, 2==MEDIUM, >=3==HIGH."""
        # 1 fastener in frame -> LOW
        res1 = severity_engine.evaluate_detections([{"class_name": "missing_fastener"}])
        assert res1.detections[0].severity == "LOW"

        # 2 fasteners in frame -> MEDIUM
        res2 = severity_engine.evaluate_detections(
            [{"class_name": "missing_fastener"}, {"class_name": "missing_fastener"}]
        )
        assert res2.detections[0].severity == "MEDIUM"
        assert res2.detections[1].severity == "MEDIUM"

        # 3 fasteners in frame -> HIGH
        res3 = severity_engine.evaluate_detections(
            [{"class_name": "missing_fastener"} for _ in range(3)]
        )
        assert res3.detections[0].severity == "HIGH"
        assert res3.detections[2].severity == "HIGH"

    def test_sleeper_misalignment_count_escalation(self, severity_engine: SeverityEngine):
        """sleeper_misalignment escalates with frame count: 1==LOW, 2==MEDIUM, >=3==HIGH."""
        res1 = severity_engine.evaluate_detections([{"class_name": "sleeper_misalignment"}])
        assert res1.detections[0].severity == "LOW"

        res3 = severity_engine.evaluate_detections(
            [{"class_name": "sleeper_misalignment"} for _ in range(4)]
        )
        assert res3.detections[0].severity == "HIGH"

    def test_obstacle_mandatory_high(self, severity_engine: SeverityEngine):
        """obstacle is mandatory HIGH regardless of size."""
        res = severity_engine.evaluate_detections(
            [{"class_name": "obstacle", "bbox_norm": [0.5, 0.5, 0.01, 0.01]}]
        )
        assert res.detections[0].severity == "HIGH"
        assert res.overall_risk_level == OverallRiskLevel.HIGH.value

    def test_track_risk_index_calculation_and_clamping(self, severity_engine: SeverityEngine):
        """Verify Track Risk Index scoring and 100.0 max clamping."""
        # Empty detections -> 0.0 TRI
        res_empty = severity_engine.evaluate_detections([])
        assert res_empty.track_risk_index == 0.0
        assert res_empty.overall_risk_level == OverallRiskLevel.NONE.value

        # 1 LOW defect (10) -> TRI 10.0
        res_low = severity_engine.evaluate_detections([{"class_name": "missing_fastener"}])
        assert res_low.track_risk_index == 10.0
        assert res_low.overall_risk_level == OverallRiskLevel.LOW.value

        # 1 MEDIUM defect (25) -> TRI 25.0
        res_med = severity_engine.evaluate_detections(
            [{"class_name": "missing_fastener"}, {"class_name": "missing_fastener"}]
        )
        # 2 MEDIUM fasteners -> 25 + 25 = 50.0
        assert res_med.track_risk_index == 50.0

        # Multiple defects exceeding 100 -> clamped to 100.0
        res_overflow = severity_engine.evaluate_detections(
            [{"class_name": "obstacle"} for _ in range(5)]
        )  # 5 * 50 = 250 -> clamped to 100.0
        assert res_overflow.track_risk_index == 100.0
        assert res_overflow.overall_risk_level == OverallRiskLevel.HIGH.value

    def test_recommended_action_strings(self, severity_engine: SeverityEngine):
        """Verify appropriate actionable dispatch recommendations are assigned."""
        res_none = severity_engine.evaluate_detections([])
        assert "nominal" in res_none.recommended_action.lower()

        res_high = severity_engine.evaluate_detections([{"class_name": "obstacle"}])
        assert "critical" in res_high.recommended_action.lower() or "emergency" in res_high.recommended_action.lower()


class TestVisionAnnotator:
    """Tests verifying OpenCV overlay rendering, badge placement, and thumbnail cropping."""

    def test_hex_to_bgr(self):
        """Verify RGB hex to OpenCV BGR color conversion."""
        assert hex_to_bgr("#D97706") == (6, 119, 217)
        assert hex_to_bgr("#DC2626") == (38, 38, 220)
        assert hex_to_bgr("invalid") == (6, 119, 217)

    def test_annotate_frame_output_structure(
        self, vision_annotator: VisionAnnotator, severity_engine: SeverityEngine
    ):
        """Annotated image preserves valid dimensions, channels, and uint8 type."""
        img = np.full((300, 400, 3), 100, dtype=np.uint8)
        analysis = severity_engine.evaluate_detections(
            [
                {
                    "class_name": "rail_crack",
                    "confidence": 0.92,
                    "bbox_xyxy": [50, 50, 150, 150],
                    "bbox_norm": [0.25, 0.33, 0.25, 0.33],
                }
            ],
            image_shape=(300, 400),
        )

        annotated = vision_annotator.annotate_frame(img, analysis, draw_hud=True)
        assert annotated.shape == (300, 400, 3)
        assert annotated.dtype == np.uint8
        # Pixel values in annotated area should change
        assert not np.array_equal(annotated, img)

    def test_extract_defect_crops(
        self, vision_annotator: VisionAnnotator, severity_engine: SeverityEngine
    ):
        """Cropped thumbnail image patches are extracted with correct dimensions."""
        img = np.full((300, 400, 3), 128, dtype=np.uint8)
        analysis = severity_engine.evaluate_detections(
            [
                {
                    "class_name": "missing_fastener",
                    "bbox_xyxy": [100, 100, 160, 160],
                    "bbox_norm": [0.325, 0.433, 0.15, 0.2],
                }
            ],
            image_shape=(300, 400),
        )

        crops = vision_annotator.extract_defect_crops(img, analysis, padding=10)
        assert len(crops) == 1
        det, crop = crops[0]
        assert det.class_name == "missing_fastener"
        assert crop.shape == (80, 80, 3)  # 60x60 box + 10px padding on each side


class TestYoloTrainer:
    """Tests verifying YOLO trainer device detection, metrics extraction, and dry-run export."""

    def test_detect_best_device(self):
        """Verify device detection logic."""
        dev = detect_best_device("auto")
        assert dev in ("0", "cpu")
        assert detect_best_device("cpu") == "cpu"

    def test_extract_evaluation_metrics(self):
        """Verify extraction from mock DetMetrics structure."""
        class MockBox:
            map50 = 0.8523
            map = 0.6912
            mp = 0.8800
            mr = 0.8100
            maps = [0.85, 0.89, 0.82, 0.0, 0.0]

        class MockVal:
            box = MockBox()
            names = {0: "rail_crack", 1: "missing_fastener", 2: "rail_corrosion"}

        metrics = extract_evaluation_metrics(MockVal())
        assert metrics["mAP50"] == 0.8523
        assert metrics["mAP50_95"] == 0.6912
        assert metrics["precision"] == 0.8800
        assert metrics["recall"] == 0.8100
        assert metrics["per_class_map"]["rail_crack"] == 0.85

    def test_dry_run_training_pipeline(self, tmp_path: Path):
        """Dry-run training generates best.pt, last.pt, and evaluation_metrics.json without GPU overhead."""
        models_dir = tmp_path / "models"
        config = YoloTrainConfig(
            models_dir=models_dir,
            dry_run=True,
        )

        report = train_railway_model(config)
        assert report["status"] == "dry_run_success"
        assert (models_dir / "best.pt").exists()
        assert (models_dir / "last.pt").exists()

        metrics_file = models_dir / "evaluation_metrics.json"
        assert metrics_file.exists()
        loaded = json.loads(metrics_file.read_text(encoding="utf-8"))
        assert loaded["metrics"]["mAP50"] > 0.0


class TestInferenceService:
    """Tests verifying end-to-end InferenceService orchestration."""

    def test_decode_image_formats(self, tmp_path: Path):
        """InferenceService correctly decodes numpy arrays, file paths, and byte buffers."""
        service = InferenceService()

        # Numpy array
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        assert service.decode_image(arr).shape == (100, 100, 3)

        # File path
        fpath = tmp_path / "test.jpg"
        cv2.imwrite(str(fpath), arr)
        assert service.decode_image(fpath).shape == (100, 100, 3)

        # Byte buffer
        _, enc = cv2.imencode(".jpg", arr)
        assert service.decode_image(enc.tobytes()).shape == (100, 100, 3)

    def test_inspect_and_crop_extraction(self):
        """InferenceService inspection returns FrameAnalysis, annotated image, and JPEG bytes."""
        service = InferenceService()
        dummy_img = np.full((200, 200, 3), 150, dtype=np.uint8)

        analysis, annotated_img, ann_bytes = service.inspect(dummy_img, draw_hud=True)
        assert isinstance(analysis, FrameAnalysis)
        assert annotated_img.shape == (200, 200, 3)
        assert isinstance(ann_bytes, bytes)
        assert len(ann_bytes) > 0

        crops = service.extract_crops(dummy_img, analysis)
        assert isinstance(crops, list)
