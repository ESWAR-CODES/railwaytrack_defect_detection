"""CLI Tool for Railway Defect Dataset Harmonization, Merging, Splitting, and Drone Augmentation.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 2: Dataset Preparation & Preprocessing Pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.engine.config_loader import load_classes_config
from backend.engine.dataset_pipeline import (
    load_and_harmonize_dataset,
    split_dataset,
    update_railway_dataset_yaml,
    write_processed_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prepare_dataset")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the dataset preparation script."""
    parser = argparse.ArgumentParser(
        description="Harmonize, merge, split, and augment railway defect datasets for YOLO training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-a",
        type=str,
        default=str(PROJECT_ROOT / "backend" / "data" / "raw" / "dataset_a"),
        help="Path to raw Dataset A root directory",
    )
    parser.add_argument(
        "--raw-b",
        type=str,
        default=str(PROJECT_ROOT / "backend" / "data" / "raw" / "dataset_b"),
        help="Path to raw Dataset B root directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "backend" / "data" / "processed"),
        help="Output destination directory for harmonized YOLO dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for dataset shuffling and sampling",
    )
    parser.add_argument(
        "--augment-factor",
        type=float,
        default=0.2,
        help="Fraction of training samples to perturb with drone-vision augmentations",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default=str(PROJECT_ROOT / "backend" / "configs" / "railway_dataset.yaml"),
        help="Path to the Ultralytics YOLO railway_dataset.yaml configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the harmonization and splitting pipeline in-memory without modifying files on disk",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the dataset preparation pipeline."""
    args = parse_arguments()

    print("\n" + "=" * 80)
    print(" AI-Based Multi-Defect Railway Monitoring and Severity Assessment")
    print(" Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3")
    print(" Phase 2: Dataset Harmonization, Merging, Splitting & Drone Augmentation")
    print("=" * 80 + "\n")

    logger.info(f"Configuration: Seed={args.seed}, AugmentFactor={args.augment_factor}, DryRun={args.dry_run}")
    logger.info(f"Input Raw A: {args.raw_a}")
    logger.info(f"Input Raw B: {args.raw_b}")
    logger.info(f"Output Directory: {args.out_dir}")

    config = load_classes_config()

    all_samples = []
    source_stats = {}

    # 1. Process Dataset A
    path_a = Path(args.raw_a)
    if path_a.exists():
        logger.info("Harmonizing Dataset A...")
        samples_a, stats_a = load_and_harmonize_dataset(path_a, source_tag="dsa", config=config)
        all_samples.extend(samples_a)
        source_stats["dataset_a"] = stats_a
    else:
        logger.warning(f"Raw Dataset A not found at: {path_a}")

    # 2. Process Dataset B
    path_b = Path(args.raw_b)
    if path_b.exists():
        logger.info("Harmonizing Dataset B...")
        samples_b, stats_b = load_and_harmonize_dataset(path_b, source_tag="dsb", config=config)
        all_samples.extend(samples_b)
        source_stats["dataset_b"] = stats_b
    else:
        logger.warning(f"Raw Dataset B not found at: {path_b}")

    total_samples = len(all_samples)
    logger.info(f"Total merged raw samples: {total_samples}")

    if total_samples == 0:
        logger.warning("No samples were discovered in the raw directories. Exiting gracefully.")
        return 0

    # 3. Deterministic 70/20/10 Split
    logger.info("Splitting dataset into Train (70%), Val (20%), and Test (10%)...")
    splits = split_dataset(
        all_samples,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=args.seed,
    )

    logger.info(
        f"Split breakdown: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}"
    )

    # 4. Write Processed Dataset & Apply Drone Augmentations
    logger.info("Writing processed samples and applying drone-vision augmentations...")
    report = write_processed_dataset(
        splits=splits,
        out_dir=Path(args.out_dir),
        config=config,
        augment_factor=args.augment_factor,
        seed=args.seed,
        dry_run=args.dry_run,
        source_stats=source_stats,
    )

    # 5. Validate / Update railway_dataset.yaml
    yaml_path = Path(args.dataset_yaml)
    if yaml_path.exists() and not args.dry_run:
        update_railway_dataset_yaml(yaml_path, Path(args.out_dir), config=config)

    # 6. Print Execution Summary Table
    print("\n" + "=" * 80)
    print(" DATASET PREPARATION SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Raw Images Ingested:   {total_samples}")
    print(f"Augmented Images Generated:  {report['augmented_sample_count']}")
    print(f"Total Processed Images:      {report['total_processed_images']}")
    print("-" * 80)
    for split_name, split_data in report["split_counts"].items():
        print(
            f"  • Split [{split_name.upper():5s}]: {split_data['total_images']:5d} images "
            f"(Original: {split_data['original_images']}, Augmented: {split_data['augmented_images']})"
        )
        print("    Class Instances:")
        for cname, count in split_data["class_instances"].items():
            print(f"      - {cname:22s}: {count:5d}")
    print("-" * 80)
    dropped_total = sum(
        sum(s.get("dropped_unmapped_classes", {}).values()) for s in source_stats.values()
    )
    print(f"Total Unmapped Labels Dropped: {dropped_total}")
    for src, st in source_stats.items():
        if st.get("dropped_unmapped_classes"):
            print(f"  [{src}]: {st['dropped_unmapped_classes']}")
    print("=" * 80 + "\n")

    logger.info("Phase 2 dataset preparation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
