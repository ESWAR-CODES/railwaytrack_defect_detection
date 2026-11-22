"""Integration and Unit Tests for Railway Defect Inspection FastAPI Service.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 4: FastAPI Inference & Inspection Backend.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import get_inference_service, set_inference_service
from backend.api.store import InspectionStore, get_inspection_store
from backend.engine.config_loader import load_classes_config
from backend.engine.inference_service import InferenceService


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """TestClient fixture with clean store state."""
    store = get_inspection_store()
    store.clear()
    with TestClient(app) as test_client:
        yield test_client
    store.clear()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a synthetic 320x320 RGB test image encoded as JPEG."""
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    # Draw simulated railway tracks and sleepers
    cv2.line(img, (80, 0), (80, 320), (180, 180, 180), 8)
    cv2.line(img, (240, 0), (240, 320), (180, 180, 180), 8)
    for y in range(40, 320, 60):
        cv2.line(img, (40, y), (280, y), (90, 60, 40), 10)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Generate a synthetic 200x200 RGB test image encoded as PNG."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", img)
    assert success
    return encoded.tobytes()


# ------------------------------------------------------------------------------
# 1. Root & Health Endpoint Tests
# ------------------------------------------------------------------------------

def test_root_endpoint(client: TestClient) -> None:
    """Verify root endpoint metadata and documentation references."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AI-Based Multi-Defect Railway Monitoring API"
    assert data["status"] == "operational"
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/api/v1/health"


def test_health_endpoint(client: TestClient) -> None:
    """Verify health endpoint reports operational state, canonical classes, and weight status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert isinstance(data["model_loaded"], bool)
    assert "classes" in data
    assert len(data["classes"]) == 5
    # Verify canonical classes present
    assert data["classes"]["0"] == "rail_crack"
    assert data["classes"]["1"] == "missing_fastener"
    assert data["classes"]["2"] == "rail_corrosion"
    assert data["classes"]["3"] == "obstacle"
    assert data["classes"]["4"] == "sleeper_misalignment"
    assert "timestamp" in data


# ------------------------------------------------------------------------------
# 2. Inspection Endpoint Tests
# ------------------------------------------------------------------------------

def test_inspect_valid_image(client: TestClient, sample_image_bytes: bytes) -> None:
    """Test full inspection pipeline with valid JPEG image upload."""
    files = {"file": ("drone_track_01.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/api/v1/inspect", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "inspection_id" in data
    assert data["filename"] == "drone_track_01.jpg"
    assert data["image_width"] == 320
    assert data["image_height"] == 320
    assert data["confidence_threshold"] == 0.25
    assert "summary" in data

    summary = data["summary"]
    assert "total_defects" in summary
    assert "track_risk_index" in summary
    assert 0.0 <= summary["track_risk_index"] <= 100.0
    assert summary["overall_risk_level"] in {"NONE", "LOW", "MEDIUM", "HIGH"}
    assert "counts_by_class" in summary
    assert "counts_by_severity" in summary

    # Annotated image base64 check
    assert data["annotated_image_base64"] is not None
    decoded = base64.b64decode(data["annotated_image_base64"])
    assert len(decoded) > 0
    # Confirm it can be decoded by OpenCV
    arr = np.frombuffer(decoded, np.uint8)
    ann_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert ann_img is not None
    assert ann_img.shape[0] == 320
    assert ann_img.shape[1] == 320


def test_inspect_png_format(client: TestClient, sample_png_bytes: bytes) -> None:
    """Test inspection with PNG format."""
    files = {"file": ("sample_frame.png", sample_png_bytes, "image/png")}
    response = client.post("/api/v1/inspect", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample_frame.png"
    assert data["image_width"] == 200
    assert data["image_height"] == 200


def test_inspect_query_parameters(client: TestClient, sample_image_bytes: bytes) -> None:
    """Test inspecting with custom confidence threshold and options via query string."""
    files = {"file": ("inspection_query.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post(
        "/api/v1/inspect?conf_threshold=0.45&include_crops=false&draw_hud=false",
        files=files,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["confidence_threshold"] == 0.45
    assert data["crops_base64"] == []


def test_inspect_form_parameters(client: TestClient, sample_image_bytes: bytes) -> None:
    """Test inspecting with custom options supplied via multipart form fields."""
    files = {"file": ("inspection_form.jpg", sample_image_bytes, "image/jpeg")}
    data_payload = {
        "conf_threshold": "0.35",
        "include_crops": "true",
        "draw_hud": "true",
    }
    response = client.post("/api/v1/inspect", files=files, data=data_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["confidence_threshold"] == 0.35


def test_inspect_confidence_clamping(client: TestClient, sample_image_bytes: bytes) -> None:
    """Test that out-of-bounds confidence thresholds clamp strictly between 0.05 and 0.95."""
    # Under-bound test
    files_low = {"file": ("test_low.jpg", sample_image_bytes, "image/jpeg")}
    res_low = client.post("/api/v1/inspect?conf_threshold=0.01", files=files_low)
    assert res_low.status_code == 200
    assert res_low.json()["confidence_threshold"] == 0.05

    # Over-bound test
    files_high = {"file": ("test_high.jpg", sample_image_bytes, "image/jpeg")}
    res_high = client.post("/api/v1/inspect?conf_threshold=0.99", files=files_high)
    assert res_high.status_code == 200
    assert res_high.json()["confidence_threshold"] == 0.95


def test_inspect_unsupported_file_extension(client: TestClient) -> None:
    """Reject unsupported file formats with 400 Bad Request."""
    files = {"file": ("report.pdf", b"%PDF-1.4...", "application/pdf")}
    response = client.post("/api/v1/inspect", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_inspect_empty_file(client: TestClient) -> None:
    """Reject empty file uploads with 400 Bad Request."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/api/v1/inspect", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_inspect_corrupt_image(client: TestClient) -> None:
    """Reject unreadable or corrupted image files with 400 Bad Request."""
    files = {"file": ("corrupt.jpg", b"This is not a real image header", "image/jpeg")}
    response = client.post("/api/v1/inspect", files=files)
    assert response.status_code == 400
    assert "Invalid or unreadable image file" in response.json()["detail"]


# ------------------------------------------------------------------------------
# 3. Metrics Endpoint Tests
# ------------------------------------------------------------------------------

def test_metrics_endpoint_when_file_exists(client: TestClient) -> None:
    """Verify metrics endpoint parses evaluation_metrics.json correctly."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()

    # If evaluation_metrics.json exists from Phase 3 training
    metrics_file = Path("backend/models/evaluation_metrics.json")
    if metrics_file.exists():
        assert data["status"] == "trained_and_evaluated"
        assert data["metrics"] is not None
        assert "mAP50" in data["metrics"]
        assert "per_class_map" in data["metrics"]
    else:
        assert data["status"] == "unavailable"


def test_metrics_endpoint_when_file_missing(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify metrics endpoint returns clean status payload when metrics file is missing."""
    monkeypatch.setattr(
        "backend.api.routes.get_metrics_path",
        lambda: Path("backend/models/non_existent_metrics.json"),
    )
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unavailable"
    assert "not been executed yet" in data["message"]


# ------------------------------------------------------------------------------
# 4. History Endpoint Tests
# ------------------------------------------------------------------------------

def test_history_endpoint_lifecycle(client: TestClient, sample_image_bytes: bytes) -> None:
    """Verify inspections are recorded and retrievable via /api/v1/history."""
    # Step 1: Initial history is empty
    res_empty = client.get("/api/v1/history")
    assert res_empty.status_code == 200
    assert res_empty.json()["total_inspections"] == 0
    assert res_empty.json()["items"] == []

    # Step 2: Perform 2 inspections
    files1 = {"file": ("track_sec1.jpg", sample_image_bytes, "image/jpeg")}
    res1 = client.post("/api/v1/inspect", files=files1)
    assert res1.status_code == 200
    id1 = res1.json()["inspection_id"]

    files2 = {"file": ("track_sec2.jpg", sample_image_bytes, "image/jpeg")}
    res2 = client.post("/api/v1/inspect", files=files2)
    assert res2.status_code == 200
    id2 = res2.json()["inspection_id"]

    # Step 3: Query history (newest first)
    res_history = client.get("/api/v1/history")
    assert res_history.status_code == 200
    hist = res_history.json()
    assert hist["total_inspections"] == 2
    assert len(hist["items"]) == 2
    assert hist["items"][0]["inspection_id"] == id2
    assert hist["items"][0]["filename"] == "track_sec2.jpg"
    assert hist["items"][1]["inspection_id"] == id1
    assert hist["items"][1]["filename"] == "track_sec1.jpg"

    # Step 4: Query with limit=1
    res_limit = client.get("/api/v1/history?limit=1")
    assert res_limit.status_code == 200
    limit_data = res_limit.json()
    assert limit_data["total_inspections"] == 2
    assert len(limit_data["items"]) == 1
    assert limit_data["items"][0]["inspection_id"] == id2


# ------------------------------------------------------------------------------
# 5. Fallback Mode & Resilience Tests
# ------------------------------------------------------------------------------

def test_inference_service_without_weights_fallback(sample_image_bytes: bytes) -> None:
    """Verify InferenceService initializes and inspects safely even if weights are absent."""
    service = InferenceService(
        model_path="backend/models/non_existent_weights.pt",
        fallback_model="non_existent_base.pt",
    )
    # Model should be None or mock, but inspect() must never crash
    analysis, annotated_np, annotated_bytes = service.inspect(sample_image_bytes)
    assert analysis.total_defects == 0
    assert analysis.track_risk_index == 0.0
    assert analysis.overall_risk_level == "NONE"
    assert annotated_np is not None
    assert len(annotated_bytes) > 0


def test_inspection_store_eviction() -> None:
    """Verify thread-safe store respects maximum capacity and eviction policy."""
    from backend.api.schemas import InspectionHistoryItem

    store = InspectionStore(max_items=3)
    for i in range(5):
        store.add(
            InspectionHistoryItem(
                inspection_id=f"id_{i}",
                filename=f"file_{i}.jpg",
                image_width=100,
                image_height=100,
                total_defects=0,
                track_risk_index=0.0,
                overall_risk_level="NONE",
                created_at="2026-11-21T00:00:00Z",
            )
        )

    assert store.count() == 3
    recent = store.list_recent(limit=10)
    assert len(recent) == 3
    assert recent[0].inspection_id == "id_4"
    assert recent[1].inspection_id == "id_3"
    assert recent[2].inspection_id == "id_2"


# ------------------------------------------------------------------------------
# 6. Sample Dataset Images Tests
# ------------------------------------------------------------------------------

def test_list_samples_endpoint(client: TestClient) -> None:
    """Verify /api/v1/samples returns available dataset and synthetic test patterns."""
    response = client.get("/api/v1/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 3
    first = data["samples"][0]
    assert "id" in first
    assert "filename" in first
    assert "split" in first
    assert "url" in first
    assert first["url"].startswith("/api/v1/samples/")


def test_get_synthetic_sample_image(client: TestClient) -> None:
    """Verify /api/v1/samples/{filename} generates and serves synthetic JPEG bytes."""
    response = client.get("/api/v1/samples/synthetic_rail_crack.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 1000

    # Verify bytes form a valid image
    arr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape == (480, 480, 3)


def test_get_sample_image_nonexistent(client: TestClient) -> None:
    """Verify /api/v1/samples/{filename} returns 404 for unknown image file."""
    response = client.get("/api/v1/samples/totally_non_existent_file_9999.jpg")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

