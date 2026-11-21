"""Computer Vision Annotator and Telemetry HUD Overlay Engine.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 3: High-Contrast Defect Overlay & Thumbnail Extraction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from backend.engine.severity_engine import DefectDetection, FrameAnalysis


def hex_to_bgr(hex_code: str) -> Tuple[int, int, int]:
    """Convert hex color string (#RRGGBB) to OpenCV BGR tuple."""
    clean = hex_code.lstrip("#")
    if len(clean) != 6:
        return (6, 119, 217)  # Fallback amber
    r = int(clean[0:2], 16)
    g = int(clean[2:4], 16)
    b = int(clean[4:6], 16)
    return (b, g, r)


class VisionAnnotator:
    """Renders high-contrast bounding boxes, label badges, telemetry HUD, and thumbnails."""

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
    ):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate_frame(
        self,
        image: np.ndarray,
        analysis: FrameAnalysis,
        draw_hud: bool = True,
        inplace: bool = False,
    ) -> np.ndarray:
        """Render defect bounding boxes, badges, and optional telemetry HUD on an image.
        
        Args:
            image: BGR numpy image array (H, W, 3).
            analysis: FrameAnalysis object with detections and TRI score.
            draw_hud: Whether to render the telemetry HUD banner across the top.
            inplace: Whether to draw directly onto the input image or on a copy.
            
        Returns:
            Annotated BGR numpy image array.
        """
        if image is None or image.size == 0:
            return image

        canvas = image if inplace else image.copy()
        img_h, img_w = canvas.shape[:2]

        # 1. Render all defect bounding boxes and badge tags
        for det in analysis.detections:
            color_bgr = hex_to_bgr(det.color_hex)
            x1, y1, x2, y2 = det.bbox_xyxy

            # Ensure integer pixel coordinates clamped to image dimensions
            ix1 = max(0, min(img_w - 1, int(round(x1))))
            iy1 = max(0, min(img_h - 1, int(round(y1))))
            ix2 = max(0, min(img_w - 1, int(round(x2))))
            iy2 = max(0, min(img_h - 1, int(round(y2))))

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            # Draw outer bounding box
            cv2.rectangle(
                canvas,
                (ix1, iy1),
                (ix2, iy2),
                color=color_bgr,
                thickness=self.box_thickness,
                lineType=cv2.LINE_AA,
            )

            # Format badge text: "class_name | SEVERITY | 0.95"
            badge_text = f"{det.class_name} | {det.severity} | {det.confidence:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(
                badge_text, self.font, self.font_scale, self.font_thickness
            )

            # Badge background positioning (place above box, or inside if too close to top)
            badge_h = text_h + baseline + 6
            badge_w = text_w + 10

            if iy1 - badge_h >= (40 if draw_hud else 0):
                bg_y1 = iy1 - badge_h
                bg_y2 = iy1
            else:
                bg_y1 = iy1
                bg_y2 = iy1 + badge_h

            bg_x1 = ix1
            bg_x2 = min(img_w, ix1 + badge_w)

            # Draw solid badge background with severity color
            cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), color=color_bgr, thickness=-1)

            # Text color: High contrast white
            text_x = bg_x1 + 5
            text_y = bg_y2 - baseline - 3
            cv2.putText(
                canvas,
                badge_text,
                (text_x, text_y),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.font_thickness,
                lineType=cv2.LINE_AA,
            )

        # 2. Render Telemetry HUD Bar
        if draw_hud and img_w >= 320 and img_h >= 120:
            self._render_telemetry_hud(canvas, analysis)

        return canvas

    def _render_telemetry_hud(self, canvas: np.ndarray, analysis: FrameAnalysis) -> None:
        """Render semi-transparent telemetry header strip across the top of the frame."""
        img_h, img_w = canvas.shape[:2]
        hud_h = 36

        # Semi-transparent dark overlay for top strip
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, hud_h), color=(15, 23, 42), thickness=-1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        # Thin bottom accent line
        accent_color = hex_to_bgr(analysis.overall_color_hex)
        cv2.line(canvas, (0, hud_h), (img_w, hud_h), color=accent_color, thickness=2, lineType=cv2.LINE_AA)

        # Left telemetry status text
        hud_text = (
            f"MBU RAIL-VISION  |  DEFECTS: {analysis.total_defects}  |  "
            f"TRI: {analysis.track_risk_index:.1f}/100"
        )
        cv2.putText(
            canvas,
            hud_text,
            (14, 23),
            self.font,
            0.45,
            (241, 245, 249),
            1,
            lineType=cv2.LINE_AA,
        )

        # Right status indicator badge
        status_text = f"STATUS: {analysis.overall_risk_level}"
        (sw, sh), sb = cv2.getTextSize(status_text, self.font, 0.45, 1)
        pill_w = sw + 16
        pill_x1 = img_w - pill_w - 12
        pill_y1 = 6
        pill_x2 = img_w - 12
        pill_y2 = hud_h - 6

        cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), color=accent_color, thickness=-1)
        cv2.putText(
            canvas,
            status_text,
            (pill_x1 + 8, pill_y2 - 6),
            self.font,
            0.42,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    def extract_defect_crops(
        self,
        image: np.ndarray,
        analysis: FrameAnalysis,
        padding: int = 12,
    ) -> List[Tuple[DefectDetection, np.ndarray]]:
        """Extract cropped thumbnail image patches for each detected railway defect.
        
        Args:
            image: Source image numpy array (H, W, 3).
            analysis: FrameAnalysis containing detected defects.
            padding: Pixel margin surrounding each bounding box.
            
        Returns:
            List of tuples: (DefectDetection, cropped_image_patch_array).
        """
        if image is None or image.size == 0 or not analysis.detections:
            return []

        img_h, img_w = image.shape[:2]
        crops: List[Tuple[DefectDetection, np.ndarray]] = []

        for det in analysis.detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            px1 = max(0, int(round(x1)) - padding)
            py1 = max(0, int(round(y1)) - padding)
            px2 = min(img_w, int(round(x2)) + padding)
            py2 = min(img_h, int(round(y2)) + padding)

            if px2 <= px1 or py2 <= py1:
                continue

            crop = image[py1:py2, px1:px2].copy()
            crops.append((det, crop))

        return crops


# Default singleton helper
_DEFAULT_ANNOTATOR: Optional[VisionAnnotator] = None


def get_vision_annotator() -> VisionAnnotator:
    """Return default singleton instance of VisionAnnotator."""
    global _DEFAULT_ANNOTATOR
    if _DEFAULT_ANNOTATOR is None:
        _DEFAULT_ANNOTATOR = VisionAnnotator()
    return _DEFAULT_ANNOTATOR


def annotate_inspection_frame(
    image: np.ndarray,
    analysis: FrameAnalysis,
    draw_hud: bool = True,
) -> np.ndarray:
    """Convenience helper to annotate an image using the default VisionAnnotator."""
    return get_vision_annotator().annotate_frame(image, analysis, draw_hud=draw_hud)
