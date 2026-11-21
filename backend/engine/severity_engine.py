"""Defect Severity Assessment and Track Risk Index (TRI) Computation Engine.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 3: Severity Assessment Engine & Feature Extraction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from backend.engine.config_loader import (
    ClassesConfig,
    SeverityLevel,
    load_classes_config,
)


class OverallRiskLevel(str, Enum):
    """Aggregate track health status categories."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


RECOMMENDED_ACTIONS: Dict[str, str] = {
    "NONE": "Track clearance nominal. No actionable railway defects detected.",
    "LOW": "Schedule routine track maintenance within standard maintenance window.",
    "MEDIUM": "Elevated track hazard detected. Dispatch maintenance crew for localized track repair.",
    "HIGH": "CRITICAL TRACK DEFECT DETECTED. Immediate speed reduction / emergency track closure required.",
}


@dataclass
class DefectDetection:
    """Detailed structural and geometric representation of an individual detected defect."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] in absolute pixel coordinates
    bbox_norm: List[float]  # [x_center, y_center, width, height] in normalized [0.0, 1.0]
    area_ratio: float       # Normalized bounding box area: width * height
    diagonal_ratio: float   # Normalized diagonal length: hypot(width, height)
    severity: str           # "LOW" | "MEDIUM" | "HIGH"
    color_hex: str          # Hexadecimal color for UI and overlay rendering
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection instance to a serializable dictionary."""
        return asdict(self)


@dataclass
class FrameAnalysis:
    """Comprehensive frame-level defect assessment and composite Track Risk Index summary."""
    detections: List[DefectDetection] = field(default_factory=list)
    total_defects: int = 0
    counts_by_class: Dict[str, int] = field(default_factory=dict)
    counts_by_severity: Dict[str, int] = field(default_factory=dict)
    track_risk_index: float = 0.0  # Clamped to [0.0, 100.0]
    overall_risk_level: str = OverallRiskLevel.NONE.value
    overall_color_hex: str = "#10B981"  # Default Emerald/Green for nominal
    recommended_action: str = RECOMMENDED_ACTIONS["NONE"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame analysis to a serializable dictionary."""
        data = asdict(self)
        data["detections"] = [d.to_dict() for d in self.detections]
        return data


class SeverityEngine:
    """Deterministic severity evaluator and composite Track Risk Index calculator."""

    def __init__(self, config: Optional[ClassesConfig] = None):
        self.config = config or load_classes_config()
        self.canonical_names = {idx: name for idx, name in self.config.classes.items()}

    def compute_geometry(
        self,
        bbox_norm: Sequence[float],
    ) -> Tuple[float, float, float, float]:
        """Compute normalized width, height, area ratio, and diagonal ratio.
        
        Args:
            bbox_norm: Normalized [x_center, y_center, width, height] in range [0.0, 1.0].
            
        Returns:
            Tuple of (width, height, area_ratio, diagonal_ratio).
        """
        w = max(0.0, min(1.0, float(bbox_norm[2])))
        h = max(0.0, min(1.0, float(bbox_norm[3])))
        area_ratio = round(w * h, 6)
        diagonal_ratio = round(math.hypot(w, h), 6)
        return w, h, area_ratio, diagonal_ratio

    def evaluate_detections(
        self,
        raw_detections: Sequence[Dict[str, Any]],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> FrameAnalysis:
        """Evaluate a list of raw YOLO detections and compute per-defect and aggregate severity.
        
        Args:
            raw_detections: List of dicts with keys:
                - 'class_id': int (canonical or raw class index)
                - 'confidence': float (detection confidence)
                - 'bbox_xyxy': [x1, y1, x2, y2] (pixel coordinates, optional if bbox_norm provided)
                - 'bbox_norm': [xc, yc, w, h] (normalized coordinates in [0, 1])
                - 'class_name': str (optional, overrides class_id if provided)
            image_shape: Optional (height, width) of the original frame for coordinate conversion.
            
        Returns:
            Populated FrameAnalysis instance.
        """
        if not raw_detections:
            return FrameAnalysis(
                detections=[],
                total_defects=0,
                counts_by_class={name: 0 for name in self.canonical_names.values()},
                counts_by_severity={"LOW": 0, "MEDIUM": 0, "HIGH": 0},
                track_risk_index=0.0,
                overall_risk_level=OverallRiskLevel.NONE.value,
                overall_color_hex="#10B981",
                recommended_action=RECOMMENDED_ACTIONS["NONE"],
            )

        # First pass: parse canonical class identities and count per-frame occurrences
        parsed_items = []
        frame_counts_by_canonical: Dict[str, int] = {name: 0 for name in self.canonical_names.values()}

        for det in raw_detections:
            # Determine canonical name
            raw_name = det.get("class_name")
            if raw_name:
                canonical_name = self.config.resolve_alias(raw_name)
                canonical_id = self.config.get_class_id(canonical_name)
            else:
                cid = int(det.get("class_id", 0))
                canonical_name = self.canonical_names.get(cid, "rail_crack")
                canonical_id = cid

            conf = float(det.get("confidence", 1.0))

            # Bounding box coordinates
            bbox_norm = det.get("bbox_norm")
            bbox_xyxy = det.get("bbox_xyxy")

            if bbox_norm is None and bbox_xyxy is not None and image_shape is not None:
                h_img, w_img = image_shape[:2]
                x1, y1, x2, y2 = bbox_xyxy
                xc = ((x1 + x2) / 2.0) / max(1, w_img)
                yc = ((y1 + y2) / 2.0) / max(1, h_img)
                bw = (x2 - x1) / max(1, w_img)
                bh = (y2 - y1) / max(1, h_img)
                bbox_norm = [xc, yc, bw, bh]

            if bbox_norm is None:
                bbox_norm = [0.5, 0.5, 0.1, 0.1]

            if bbox_xyxy is None and image_shape is not None:
                h_img, w_img = image_shape[:2]
                xc, yc, bw, bh = bbox_norm
                x1 = (xc - bw / 2.0) * w_img
                y1 = (yc - bh / 2.0) * h_img
                x2 = (xc + bw / 2.0) * w_img
                y2 = (yc + bh / 2.0) * h_img
                bbox_xyxy = [x1, y1, x2, y2]
            elif bbox_xyxy is None:
                bbox_xyxy = [0.0, 0.0, 0.0, 0.0]

            frame_counts_by_canonical[canonical_name] += 1
            parsed_items.append((canonical_id, canonical_name, conf, bbox_xyxy, bbox_norm))

        # Second pass: compute deterministic severity per detection
        detections: List[DefectDetection] = []
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

        for cid, cname, conf, b_xyxy, b_norm in parsed_items:
            w, h, area_ratio, diagonal_ratio = self.compute_geometry(b_norm)

            # Evaluate severity according to canonical rules
            if cname == "obstacle":
                # Mandatory HIGH
                sev_level = SeverityLevel.HIGH
            elif cname in ("missing_fastener", "sleeper_misalignment"):
                # Count-based escalation based on frame occurrence
                cluster_count = frame_counts_by_canonical[cname]
                sev_level = self.config.evaluate_severity(cname, cluster_count)
            elif cname == "rail_crack":
                # Geometric metric: diagonal_ratio
                sev_level = self.config.evaluate_severity(cname, diagonal_ratio)
            elif cname == "rail_corrosion":
                # Geometric metric: area_ratio
                sev_level = self.config.evaluate_severity(cname, area_ratio)
            else:
                sev_level = SeverityLevel.LOW

            sev_str = sev_level.value
            color_hex = self.config.get_severity_color(sev_level)
            action = RECOMMENDED_ACTIONS.get(sev_str, RECOMMENDED_ACTIONS["LOW"])

            severity_counts[sev_str] = severity_counts.get(sev_str, 0) + 1

            detections.append(
                DefectDetection(
                    class_id=cid,
                    class_name=cname,
                    confidence=round(conf, 4),
                    bbox_xyxy=[round(float(coord), 2) for coord in b_xyxy],
                    bbox_norm=[round(float(coord), 6) for coord in b_norm],
                    area_ratio=area_ratio,
                    diagonal_ratio=diagonal_ratio,
                    severity=sev_str,
                    color_hex=color_hex,
                    recommended_action=action,
                )
            )

        # Calculate composite Track Risk Index (TRI)
        all_severities = [d.severity for d in detections]
        raw_tri = self.config.calculate_track_risk_index(all_severities)
        clamped_tri = min(100.0, max(0.0, round(raw_tri, 1)))

        # Determine overall risk status
        if not detections:
            overall_status = OverallRiskLevel.NONE.value
            overall_color = "#10B981"
            overall_action = RECOMMENDED_ACTIONS["NONE"]
        elif severity_counts.get("HIGH", 0) > 0 or clamped_tri >= 50.0:
            overall_status = OverallRiskLevel.HIGH.value
            overall_color = self.config.get_severity_color("HIGH")
            overall_action = RECOMMENDED_ACTIONS["HIGH"]
        elif severity_counts.get("MEDIUM", 0) > 0 or clamped_tri >= 25.0:
            overall_status = OverallRiskLevel.MEDIUM.value
            overall_color = self.config.get_severity_color("MEDIUM")
            overall_action = RECOMMENDED_ACTIONS["MEDIUM"]
        else:
            overall_status = OverallRiskLevel.LOW.value
            overall_color = self.config.get_severity_color("LOW")
            overall_action = RECOMMENDED_ACTIONS["LOW"]

        return FrameAnalysis(
            detections=detections,
            total_defects=len(detections),
            counts_by_class=frame_counts_by_canonical,
            counts_by_severity=severity_counts,
            track_risk_index=clamped_tri,
            overall_risk_level=overall_status,
            overall_color_hex=overall_color,
            recommended_action=overall_action,
        )


# Global default instance helper
_DEFAULT_SEVERITY_ENGINE: Optional[SeverityEngine] = None


def get_severity_engine() -> SeverityEngine:
    """Return singleton instance of SeverityEngine."""
    global _DEFAULT_SEVERITY_ENGINE
    if _DEFAULT_SEVERITY_ENGINE is None:
        _DEFAULT_SEVERITY_ENGINE = SeverityEngine()
    return _DEFAULT_SEVERITY_ENGINE


def evaluate_frame_defects(
    raw_detections: Sequence[Dict[str, Any]],
    image_shape: Optional[Tuple[int, int]] = None,
) -> FrameAnalysis:
    """Convenience helper to evaluate frame defects with default configuration."""
    return get_severity_engine().evaluate_detections(raw_detections, image_shape)
