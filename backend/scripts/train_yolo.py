"""CLI Script to Train, Evaluate, and Export YOLO Railway Defect Detection Models.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 3: YOLO Model Training & Checkpoint Export.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.engine.yolo_trainer import YoloTrainConfig, train_railway_model

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_yolo")


def parse_arguments() -> argparse.Namespace:
    """Parse command line flags for YOLO model training."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Ultralytics YOLO models on railway defect datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base pretrained YOLO model checkpoint (e.g. yolo11n.pt, yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default=str(PROJECT_ROOT / "backend" / "configs" / "railway_dataset.yaml"),
        help="Path to Ultralytics dataset YAML configuration",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for gradient descent",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image resolution (height and width) for model input",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('auto', 'cpu', '0', '1')",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use for training (useful for fast smoke-testing on CPU)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(PROJECT_ROOT / "backend" / "models"),
        help="Directory to save best.pt and evaluation_metrics.json",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(PROJECT_ROOT / "backend" / "runs"),
        help="Root directory for YOLO runs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="railway_yolo",
        help="Experiment name subfolder",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the training run and generate mock metrics/weights for testing",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the YOLO training and evaluation pipeline."""
    args = parse_arguments()

    print("\n" + "=" * 80)
    print(" AI-Based Multi-Defect Railway Monitoring and Severity Assessment")
    print(" Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3")
    print(" Phase 3: YOLO Model Training & Checkpoint Export")
    print("=" * 80 + "\n")

    train_config = YoloTrainConfig(
        model_name=args.model,
        dataset_yaml=Path(args.dataset_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=Path(args.project),
        name=args.name,
        models_dir=Path(args.models_dir),
        fraction=args.fraction,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    logger.info(
        f"Training Params: Model={args.model}, Epochs={args.epochs}, Batch={args.batch}, "
        f"ImgSz={args.imgsz}, Fraction={args.fraction}, Device={args.device}"
    )

    report = train_railway_model(train_config)

    print("\n" + "=" * 80)
    print(" YOLO TRAINING & EVALUATION REPORT")
    print("=" * 80)
    print(f"Status:       {report.get('status')}")
    print(f"Model Path:   {report.get('model_path')}")
    print(f"Device Used:  {report.get('device')}")
    print("-" * 80)
    metrics = report.get("metrics", {})
    print(f"  • mAP@0.50:       {metrics.get('mAP50', 0.0):.4f}")
    print(f"  • mAP@0.50:0.95:  {metrics.get('mAP50_95', 0.0):.4f}")
    print(f"  • Precision:      {metrics.get('precision', 0.0):.4f}")
    print(f"  • Recall:         {metrics.get('recall', 0.0):.4f}")
    if metrics.get("per_class_map"):
        print("  • Class Breakdown (mAP@0.50):")
        for cname, score in metrics["per_class_map"].items():
            print(f"      - {cname:24s}: {score:.4f}")
    print("=" * 80 + "\n")

    logger.info("YOLO training and evaluation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
