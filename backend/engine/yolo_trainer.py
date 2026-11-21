"""Ultralytics YOLO Training, Evaluation, and Checkpoint Export Engine.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 3: YOLO Model Training & Performance Validation.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger("yolo_trainer")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class YoloTrainConfig:
    """Hyperparameter and training configuration for YOLO railway model."""
    model_name: str = "yolo11n.pt"
    dataset_yaml: Path = Path("backend/configs/railway_dataset.yaml")
    epochs: int = 50
    batch: int = 16
    imgsz: int = 640
    device: str = "auto"
    project: Path = Path("backend/runs")
    name: str = "railway_yolo"
    models_dir: Path = Path("backend/models")
    fraction: float = 1.0
    seed: int = 42
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        d = asdict(self)
        d["dataset_yaml"] = str(self.dataset_yaml)
        d["project"] = str(self.project)
        d["models_dir"] = str(self.models_dir)
        return d


def resolve_dataset_yaml_for_ultralytics(dataset_yaml_path: Path) -> Path:
    """Read dataset YAML and ensure the root dataset path is resolved to an absolute directory.
    
    This preserves the relative '../data/processed' in the original configuration while
    ensuring Ultralytics check_det_dataset works reliably across any execution working directory.
    """
    dataset_yaml_path = Path(dataset_yaml_path).resolve()
    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_path = data.get("path", "")
    resolved_dir = (dataset_yaml_path.parent / raw_path).resolve()

    # Check if images directory exists under resolved path
    if not (resolved_dir / "images").exists():
        fallback_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
        if (fallback_dir / "images").exists():
            resolved_dir = fallback_dir

    data["path"] = str(resolved_dir).replace("\\", "/")

    resolved_yaml = dataset_yaml_path.parent / "resolved_railway_dataset.yaml"
    with open(resolved_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)

    return resolved_yaml


def detect_best_device(requested_device: str = "auto") -> str:
    """Determine the optimal compute device (CUDA GPU if available, else CPU)."""
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA GPU detected: {gpu_name}. Selecting device='0'.")
        return "0"
    else:
        logger.info("No CUDA device detected. Selecting device='cpu'.")
        return "cpu"


def extract_evaluation_metrics(val_results: Any) -> Dict[str, Any]:
    """Safely extract mAP50, mAP50-95, precision, and recall from Ultralytics val results."""
    box = getattr(val_results, "box", None)
    results_dict = getattr(val_results, "results_dict", {})

    # Extract primary metrics with fallback across Ultralytics versions
    map50 = getattr(box, "map50", None)
    if map50 is None:
        map50 = results_dict.get("metrics/mAP50(B)", 0.0)

    map50_95 = getattr(box, "map", None)
    if map50_95 is None:
        map50_95 = results_dict.get("metrics/mAP50-95(B)", 0.0)

    mp = getattr(box, "mp", None)
    if mp is None:
        mp = results_dict.get("metrics/precision(B)", 0.0)

    mr = getattr(box, "mr", None)
    if mr is None:
        mr = results_dict.get("metrics/recall(B)", 0.0)

    # Class-wise mAP50 if available
    class_names = getattr(val_results, "names", {})
    class_maps = getattr(box, "maps", None)
    per_class = {}
    if class_maps is not None and hasattr(class_maps, "__len__"):
        for idx, score in enumerate(class_maps):
            cname = class_names.get(idx, f"class_{idx}")
            per_class[cname] = round(float(score), 4)

    return {
        "mAP50": round(float(map50), 4),
        "mAP50_95": round(float(map50_95), 4),
        "precision": round(float(mp), 4),
        "recall": round(float(mr), 4),
        "per_class_map": per_class,
    }


def train_railway_model(config: YoloTrainConfig) -> Dict[str, Any]:
    """Train YOLO model on the railway dataset, evaluate on test split, and export best.pt."""
    models_dir = Path(config.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = models_dir / "evaluation_metrics.json"

    # 1. Handle Dry-Run Simulation Mode
    if config.dry_run:
        logger.info("Executing YOLO trainer in DRY-RUN mode.")
        mock_best_pt = models_dir / "best.pt"
        if not mock_best_pt.exists():
            mock_best_pt.write_text("# MOCK YOLO WEIGHTS FOR TESTING\n", encoding="utf-8")

        mock_last_pt = models_dir / "last.pt"
        if not mock_last_pt.exists():
            mock_last_pt.write_text("# MOCK YOLO LAST WEIGHTS FOR TESTING\n", encoding="utf-8")

        mock_report = {
            "timestamp": datetime.now().isoformat(),
            "status": "dry_run_success",
            "model_path": str(mock_best_pt),
            "device": "dry_run",
            "training_parameters": config.to_dict(),
            "metrics": {
                "mAP50": 0.8950,
                "mAP50_95": 0.7420,
                "precision": 0.8710,
                "recall": 0.8430,
                "per_class_map": {
                    "rail_crack": 0.8850,
                    "missing_fastener": 0.9120,
                    "rail_corrosion": 0.8880,
                    "obstacle": 0.0000,
                    "sleeper_misalignment": 0.0000,
                },
            },
        }

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(mock_report, f, indent=2)
        logger.info(f"Mock evaluation metrics saved to {metrics_file}")
        return mock_report

    if YOLO is None:
        raise ImportError("Ultralytics is required for model training. Run `pip install ultralytics`.")

    device = detect_best_device(config.device)
    dataset_yaml = Path(config.dataset_yaml).resolve()

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML config not found at: {dataset_yaml}")

    resolved_dataset_yaml = resolve_dataset_yaml_for_ultralytics(dataset_yaml)

    logger.info(f"Initializing YOLO model: {config.model_name}")
    model = YOLO(config.model_name)

    project_dir = Path(config.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # 2. Execute Training
    logger.info(
        f"Starting YOLO training for {config.epochs} epochs, imgsz={config.imgsz}, "
        f"batch={config.batch}, device={device}, fraction={config.fraction}"
    )

    train_results = model.train(
        data=str(resolved_dataset_yaml),
        epochs=config.epochs,
        batch=config.batch,
        imgsz=config.imgsz,
        device=device,
        project=str(project_dir),
        name=config.name,
        fraction=config.fraction,
        seed=config.seed,
        exist_ok=True,
        verbose=True,
    )

    # 3. Locate & Copy Checkpoint Artifacts
    run_dir = project_dir / config.name
    weights_dir = run_dir / "weights"
    best_pt_source = weights_dir / "best.pt"
    last_pt_source = weights_dir / "last.pt"

    dest_best_pt = models_dir / "best.pt"
    dest_last_pt = models_dir / "last.pt"

    if best_pt_source.exists():
        shutil.copy2(best_pt_source, dest_best_pt)
        logger.info(f"Copied best model checkpoint to: {dest_best_pt}")
    else:
        # Fallback to model weights in memory
        try:
            model.save(str(dest_best_pt))
        except Exception as e:
            logger.warning(f"Could not save best.pt: {e}")

    if last_pt_source.exists():
        shutil.copy2(last_pt_source, dest_last_pt)
        logger.info(f"Copied last model checkpoint to: {dest_last_pt}")

    # 4. Evaluate Model on Test Split
    logger.info("Evaluating model on test split...")
    try:
        val_results = model.val(
            data=str(resolved_dataset_yaml),
            split="test",
            imgsz=config.imgsz,
            device=device,
        )
        extracted_metrics = extract_evaluation_metrics(val_results)
    except Exception as e:
        logger.warning(f"Test split validation encountered an issue ({e}). Falling back to val split...")
        try:
            val_results = model.val(
                data=str(resolved_dataset_yaml),
                split="val",
                imgsz=config.imgsz,
                device=device,
            )
            extracted_metrics = extract_evaluation_metrics(val_results)
        except Exception as e2:
            logger.error(f"Validation failed: {e2}")
            extracted_metrics = {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}

    # 5. Write Structured Evaluation Report
    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "trained_and_evaluated",
        "model_path": str(dest_best_pt),
        "device": device,
        "training_parameters": config.to_dict(),
        "metrics": extracted_metrics,
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation metrics report written to: {metrics_file}")

    return report
