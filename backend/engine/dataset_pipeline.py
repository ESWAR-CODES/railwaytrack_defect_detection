"""Dataset Harmonization, Augmentation, and Splitting Pipeline.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 2: Dataset Harmonization, Merging, Splitting & Drone Augmentation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import yaml

from backend.engine.config_loader import (
    ClassesConfig,
    get_default_config_path,
    load_classes_config,
)

logger = logging.getLogger("dataset_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class BBox:
    """Normalized YOLO bounding box: class_id, x_center, y_center, width, height."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_line(self) -> str:
        """Format as a standard YOLO text line."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_albumentations_bbox(self) -> List[float]:
        """Convert to Albumentations [x_center, y_center, width, height] format with strict [0, 1] bounds."""
        x1 = self.x_center - self.width / 2.0
        y1 = self.y_center - self.height / 2.0
        x2 = self.x_center + self.width / 2.0
        y2 = self.y_center + self.height / 2.0

        if x1 >= 1e-5 and y1 >= 1e-5 and x2 <= 1.0 - 1e-5 and y2 <= 1.0 - 1e-5:
            return [self.x_center, self.y_center, self.width, self.height]

        x1_safe = max(1e-4, min(1.0 - 2e-4, x1))
        y1_safe = max(1e-4, min(1.0 - 2e-4, y1))
        x2_safe = max(x1_safe + 1e-4, min(1.0 - 1e-4, x2))
        y2_safe = max(y1_safe + 1e-4, min(1.0 - 1e-4, y2))
        w = max(1e-4, x2_safe - x1_safe)
        h = max(1e-4, y2_safe - y1_safe)
        xc = (x1_safe + x2_safe) / 2.0
        yc = (y1_safe + y2_safe) / 2.0
        return [xc, yc, w, h]


@dataclass
class SampleItem:
    """Represents an image and its set of normalized canonical YOLO bounding boxes."""
    image_path: Path
    source_tag: str
    bboxes: List[BBox] = field(default_factory=list)
    label_path: Optional[Path] = None
    original_stem: str = ""

    def __post_init__(self):
        if not self.original_stem:
            self.original_stem = self.image_path.stem


def clamp_and_validate_bbox(
    class_id: int,
    x_c: float,
    y_c: float,
    w: float,
    h: float,
    min_size: float = 1e-4,
) -> Optional[BBox]:
    """Validate and clamp bounding box coordinates strictly to [0.0, 1.0].
    
    Returns None if the bounding box has degenerate dimensions or is invalid.
    """
    try:
        x_c = float(x_c)
        y_c = float(y_c)
        w = float(w)
        h = float(h)
        class_id = int(class_id)
    except (ValueError, TypeError):
        return None

    if any(math.isnan(v) or math.isinf(v) for v in (x_c, y_c, w, h)):
        return None

    if w <= 0.0 or h <= 0.0:
        return None

    # Calculate corner coordinates
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0

    # Clamp corners to [0.0, 1.0]
    x1_clamped = max(0.0, min(1.0, x1))
    y1_clamped = max(0.0, min(1.0, y1))
    x2_clamped = max(0.0, min(1.0, x2))
    y2_clamped = max(0.0, min(1.0, y2))

    clamped_w = x2_clamped - x1_clamped
    clamped_h = y2_clamped - y1_clamped

    if clamped_w < min_size or clamped_h < min_size:
        return None

    new_xc = (x1_clamped + x2_clamped) / 2.0
    new_yc = (y1_clamped + y2_clamped) / 2.0

    # Guard boundaries
    new_xc = max(0.0, min(1.0, new_xc))
    new_yc = max(0.0, min(1.0, new_yc))
    clamped_w = max(min_size, min(1.0, clamped_w))
    clamped_h = max(min_size, min(1.0, clamped_h))

    xc_rounded = round(new_xc, 6)
    yc_rounded = round(new_yc, 6)
    w_rounded = round(clamped_w, 6)
    h_rounded = round(clamped_h, 6)

    # Prevent rounding from pushing edge coordinates beyond [0.0, 1.0]
    if xc_rounded - w_rounded / 2.0 < 0.0:
        w_rounded = max(min_size, round(2.0 * xc_rounded, 6))
    if xc_rounded + w_rounded / 2.0 > 1.0:
        w_rounded = max(min_size, round(2.0 * (1.0 - xc_rounded), 6))
    if yc_rounded - h_rounded / 2.0 < 0.0:
        h_rounded = max(min_size, round(2.0 * yc_rounded, 6))
    if yc_rounded + h_rounded / 2.0 > 1.0:
        h_rounded = max(min_size, round(2.0 * (1.0 - yc_rounded), 6))

    return BBox(
        class_id=class_id,
        x_center=xc_rounded,
        y_center=yc_rounded,
        width=w_rounded,
        height=h_rounded,
    )


def find_dataset_root(raw_dir: Union[str, Path]) -> Path:
    """Detect if raw_dir has nested subfolders (e.g. from an extracted ZIP archive).
    
    Returns the deepest directory containing data.yaml or train/valid/test directories.
    """
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.exists():
        return raw_dir

    # Check direct directory
    if (raw_dir / "data.yaml").exists() or (raw_dir / "train").exists() or (raw_dir / "images").exists():
        return raw_dir

    # Check 1 level down
    for sub in raw_dir.iterdir():
        if sub.is_dir():
            if (sub / "data.yaml").exists() or (sub / "train").exists() or (sub / "images").exists():
                return sub

    # Check 2 levels down
    for sub in raw_dir.glob("*/*"):
        if sub.is_dir():
            if (sub / "data.yaml").exists() or (sub / "train").exists() or (sub / "images").exists():
                return sub

    return raw_dir


def parse_dataset_classes(data_yaml_path: Optional[Union[str, Path]]) -> Dict[int, str]:
    """Parse class names dictionary from a raw dataset's data.yaml file."""
    if not data_yaml_path:
        return {}

    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.exists():
        return {}

    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to read data.yaml at {data_yaml_path}: {e}")
        return {}

    if not isinstance(data, dict):
        return {}

    names = data.get("names", [])
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    return {}


def find_image_label_pairs(root_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    """Find all image files in a dataset directory paired with their corresponding .txt label files.
    
    Supports:
    - Standard YOLO split folders: split/images/ and split/labels/
    - Flat folders with images and labels side-by-side
    - Nested directory scans
    """
    pairs: List[Tuple[Path, Optional[Path]]] = []
    seen_images: Set[Path] = set()

    # Search for all image files recursively
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if file_path in seen_images:
            continue

        seen_images.add(file_path)

        # 1. Look in corresponding labels folder (standard YOLO: path/.../images/... -> path/.../labels/...)
        candidate_label: Optional[Path] = None
        parts = list(file_path.parts)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
            label_in_labels_dir = Path(*parts).with_suffix(".txt")
            if label_in_labels_dir.exists():
                candidate_label = label_in_labels_dir

        # 2. Look in same directory as the image
        if candidate_label is None:
            same_dir_label = file_path.with_suffix(".txt")
            if same_dir_label.exists():
                candidate_label = same_dir_label

        # 3. Look in sibling labels directory
        if candidate_label is None:
            sibling_labels_dir = file_path.parent.parent / "labels" / f"{file_path.stem}.txt"
            if sibling_labels_dir.exists():
                candidate_label = sibling_labels_dir

        pairs.append((file_path, candidate_label))

    return pairs


def load_coco_annotations_fallback(
    coco_json_path: Path,
    images_dir: Path,
    source_tag: str,
    config: ClassesConfig,
) -> Tuple[List[SampleItem], Dict[str, Any]]:
    """Fallback loader for COCO JSON formatted datasets (e.g. _annotations.coco.json)."""
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Categories
    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    # Images map
    images_info = {img["id"]: img for img in coco_data.get("images", [])}

    # Group annotations by image_id
    img_bboxes: Dict[int, List[BBox]] = {img_id: [] for img_id in images_info}
    dropped_classes: Dict[str, int] = {}
    malformed_lines = 0

    for ann in coco_data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in images_info:
            continue

        img_meta = images_info[img_id]
        im_w = float(img_meta.get("width", 1.0))
        im_h = float(img_meta.get("height", 1.0))
        if im_w <= 0 or im_h <= 0:
            continue

        cat_name = categories.get(ann.get("category_id"))
        if not cat_name:
            continue

        try:
            canonical_name = config.resolve_alias(cat_name)
            canonical_id = config.get_class_id(canonical_name)
        except ValueError:
            dropped_classes[cat_name] = dropped_classes.get(cat_name, 0) + 1
            continue

        bbox_raw = ann.get("bbox", [])
        if len(bbox_raw) < 4:
            malformed_lines += 1
            continue

        # COCO bbox: [x_min, y_min, width, height] in pixels
        x_min, y_min, w_px, h_px = bbox_raw[:4]
        xc_norm = (x_min + w_px / 2.0) / im_w
        yc_norm = (y_min + h_px / 2.0) / im_h
        w_norm = w_px / im_w
        h_norm = h_px / im_h

        clamped = clamp_and_validate_bbox(canonical_id, xc_norm, yc_norm, w_norm, h_norm)
        if clamped:
            img_bboxes[img_id].append(clamped)

    samples: List[SampleItem] = []
    for img_id, img_meta in images_info.items():
        file_name = img_meta.get("file_name", "")
        img_path = images_dir / file_name
        if not img_path.exists():
            # Search recursively
            found = list(images_dir.rglob(file_name))
            if found:
                img_path = found[0]
            else:
                continue

        samples.append(
            SampleItem(
                image_path=img_path,
                source_tag=source_tag,
                bboxes=img_bboxes.get(img_id, []),
                label_path=coco_json_path,
                original_stem=img_path.stem,
            )
        )

    stats = {
        "total_images": len(samples),
        "dropped_unmapped_classes": dropped_classes,
        "malformed_lines_skipped": malformed_lines,
    }
    return samples, stats


def load_and_harmonize_dataset(
    raw_dir: Union[str, Path],
    source_tag: str,
    config: Optional[ClassesConfig] = None,
) -> Tuple[List[SampleItem], Dict[str, Any]]:
    """Load, inspect, and harmonize a raw dataset into canonical 5-class YOLO format.
    
    Returns:
        (samples, stats_dict)
    """
    if config is None:
        config = load_classes_config()

    root_dir = find_dataset_root(raw_dir)
    logger.info(f"Loading dataset [{source_tag}] from resolved root: {root_dir}")

    # Fallback to COCO if no .txt files are present but COCO JSON exists
    coco_json_candidates = list(root_dir.rglob("_annotations.coco.json")) + list(
        root_dir.rglob("annotations.json")
    )
    yolo_pairs = find_image_label_pairs(root_dir)

    has_yolo_labels = any(lbl is not None for _, lbl in yolo_pairs)

    if not has_yolo_labels and coco_json_candidates:
        logger.info(f"YOLO labels not found, falling back to COCO JSON: {coco_json_candidates[0]}")
        return load_coco_annotations_fallback(
            coco_json_candidates[0], root_dir, source_tag, config
        )

    # Standard YOLO path
    data_yaml_path = root_dir / "data.yaml"
    raw_classes = parse_dataset_classes(data_yaml_path)
    logger.info(f"Parsed {len(raw_classes)} raw class names from {data_yaml_path}")

    samples: List[SampleItem] = []
    dropped_classes: Dict[str, int] = {}
    malformed_lines_skipped = 0
    total_boxes_loaded = 0
    class_instance_counts: Dict[str, int] = {name: 0 for name in config.classes.values()}

    for img_path, lbl_path in yolo_pairs:
        valid_bboxes: List[BBox] = []

        if lbl_path and lbl_path.exists():
            try:
                content = lbl_path.read_text(encoding="utf-8", errors="ignore")
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        malformed_lines_skipped += 1
                        continue

                    try:
                        raw_cid = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                    except (ValueError, TypeError):
                        malformed_lines_skipped += 1
                        continue

                    raw_cname = raw_classes.get(raw_cid, str(raw_cid))

                    # Resolve alias to canonical class
                    try:
                        canonical_name = config.resolve_alias(raw_cname)
                        canonical_id = config.get_class_id(canonical_name)
                    except ValueError:
                        # Unmapped or irrelevant class (e.g. fastener_present)
                        dropped_classes[raw_cname] = dropped_classes.get(raw_cname, 0) + 1
                        continue

                    clamped = clamp_and_validate_bbox(canonical_id, xc, yc, w, h)
                    if clamped is None:
                        malformed_lines_skipped += 1
                        continue

                    valid_bboxes.append(clamped)
                    class_instance_counts[canonical_name] += 1
                    total_boxes_loaded += 1

            except Exception as e:
                logger.warning(f"Error reading label file {lbl_path}: {e}")

        samples.append(
            SampleItem(
                image_path=img_path,
                source_tag=source_tag,
                bboxes=valid_bboxes,
                label_path=lbl_path,
                original_stem=img_path.stem,
            )
        )

    stats = {
        "source_tag": source_tag,
        "raw_root": str(root_dir),
        "total_images_found": len(samples),
        "total_labels_found": sum(1 for _, lbl in yolo_pairs if lbl is not None),
        "total_boxes_loaded": total_boxes_loaded,
        "dropped_unmapped_classes": dropped_classes,
        "malformed_lines_skipped": malformed_lines_skipped,
        "class_instance_counts": class_instance_counts,
    }

    logger.info(
        f"[{source_tag}] Successfully loaded {len(samples)} images, {total_boxes_loaded} valid boxes, "
        f"dropped unmapped: {dropped_classes}"
    )
    return samples, stats


def split_dataset(
    samples: List[SampleItem],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[SampleItem]]:
    """Deterministically split samples into train (70%), val (20%), and test (10%)."""
    if not samples:
        return {"train": [], "val": [], "test": []}

    total = len(samples)
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    # Deterministic shuffle
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_train = int(round(total * train_ratio))
    n_val = int(round(total * val_ratio))
    n_test = total - n_train - n_val

    # Edge cases for small sample counts (e.g. unit tests)
    if total >= 3:
        if n_train == 0 and train_ratio > 0:
            n_train = 1
        if n_val == 0 and val_ratio > 0:
            n_val = 1
        if n_test == 0 and test_ratio > 0:
            n_test = 1
        # Re-balance to sum to total
        diff = (n_train + n_val + n_test) - total
        if diff > 0 and n_train > diff:
            n_train -= diff

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train : n_train + n_val]
    test_set = shuffled[n_train + n_val :]

    return {"train": train_set, "val": val_set, "test": test_set}


def create_drone_augmentation_pipeline() -> A.Compose:
    """Create an Albumentations composition for drone-vision perturbations.
    
    Includes:
    - Safe rotation within ±15 degrees with reflection border
    - Motion blur simulating drone vibration / transit
    - Illumination changes (brightness/contrast)
    - Atmospheric weather effects (fog/rain)
    - YOLO bbox bounding parameter enforcement (min_visibility=0.3)
    """
    return A.Compose(
        [
            A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_REFLECT_101),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.OneOf(
                [
                    A.RandomFog(p=0.5),
                    A.RandomRain(p=0.5),
                ],
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def augment_sample(
    image: np.ndarray,
    bboxes: List[BBox],
    augmentor: A.Compose,
) -> Tuple[np.ndarray, List[BBox]]:
    """Apply drone vision augmentations to an image and its bounding boxes."""
    if image is None or image.size == 0:
        return image, bboxes

    alb_bboxes = [b.to_albumentations_bbox() for b in bboxes]
    class_labels = [b.class_id for b in bboxes]

    try:
        transformed = augmentor(image=image, bboxes=alb_bboxes, class_labels=class_labels)
        aug_image = transformed["image"]
        aug_bboxes_raw = transformed["bboxes"]
        aug_labels_raw = transformed["class_labels"]

        valid_aug_bboxes: List[BBox] = []
        for raw_box, cid in zip(aug_bboxes_raw, aug_labels_raw):
            clamped = clamp_and_validate_bbox(
                int(cid), raw_box[0], raw_box[1], raw_box[2], raw_box[3]
            )
            if clamped:
                valid_aug_bboxes.append(clamped)

        return aug_image, valid_aug_bboxes
    except Exception as e:
        logger.warning(f"Augmentation skipped for sample due to: {e}")
        return image, bboxes


def make_unique_stem(existing_stems: Set[str], prefix: str, original_stem: str) -> str:
    """Generate a collision-free filename stem prefixed with source tag."""
    base = f"{prefix}_{original_stem}"
    if base not in existing_stems:
        existing_stems.add(base)
        return base

    idx = 1
    while f"{base}_{idx}" in existing_stems:
        idx += 1
    unique_stem = f"{base}_{idx}"
    existing_stems.add(unique_stem)
    return unique_stem


def write_processed_dataset(
    splits: Dict[str, List[SampleItem]],
    out_dir: Union[str, Path],
    config: Optional[ClassesConfig] = None,
    augment_factor: float = 0.2,
    seed: int = 42,
    dry_run: bool = False,
    source_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write harmonized and split dataset to backend/data/processed with optional augmentation.
    
    Generates backend/data/processed/dataset_report.json and validates YOLO directory structure.
    """
    if config is None:
        config = load_classes_config()

    out_dir = Path(out_dir).resolve()
    logger.info(f"Target processed directory: {out_dir} (dry_run={dry_run})")

    # Destination directories
    split_names = ["train", "val", "test"]
    if not dry_run:
        for split in split_names:
            (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    augmentor = create_drone_augmentation_pipeline()
    rng = random.Random(seed)

    split_reports: Dict[str, Any] = {}
    used_stems: Set[str] = set()
    total_processed_images = 0
    total_augmented_images = 0

    canonical_names = {idx: name for idx, name in config.classes.items()}

    for split in split_names:
        sample_list = splits.get(split, [])
        class_instances = {name: 0 for name in canonical_names.values()}
        split_orig_count = 0
        split_aug_count = 0

        for sample in sample_list:
            unique_stem = make_unique_stem(used_stems, sample.source_tag, sample.original_stem)
            ext = sample.image_path.suffix.lower() if sample.image_path.suffix else ".jpg"

            dest_img = out_dir / "images" / split / f"{unique_stem}{ext}"
            dest_lbl = out_dir / "labels" / split / f"{unique_stem}.txt"

            if not dry_run:
                # Copy image
                try:
                    shutil.copy2(sample.image_path, dest_img)
                except Exception as e:
                    logger.error(f"Failed to copy image {sample.image_path} to {dest_img}: {e}")
                    continue

                # Write label file
                label_lines = [b.to_yolo_line() for b in sample.bboxes]
                dest_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

            split_orig_count += 1
            for b in sample.bboxes:
                cname = canonical_names.get(b.class_id, "unknown")
                class_instances[cname] = class_instances.get(cname, 0) + 1

        # Apply augmentations strictly to the 'train' split
        if split == "train" and augment_factor > 0 and sample_list:
            num_to_augment = (
                int(round(len(sample_list) * augment_factor))
                if augment_factor <= 1.0
                else int(augment_factor)
            )
            # Select samples to augment
            candidates = sample_list if num_to_augment >= len(sample_list) else rng.sample(sample_list, num_to_augment)

            for idx, sample in enumerate(candidates, start=1):
                ext = sample.image_path.suffix.lower() if sample.image_path.suffix else ".jpg"
                aug_stem = f"{sample.source_tag}_{sample.original_stem}_aug{idx}"
                aug_img_path = out_dir / "images" / "train" / f"{aug_stem}{ext}"
                aug_lbl_path = out_dir / "labels" / "train" / f"{aug_stem}.txt"

                if not dry_run:
                    img_mat = cv2.imread(str(sample.image_path))
                    if img_mat is None:
                        logger.warning(f"Could not load image for augmentation: {sample.image_path}")
                        continue

                    aug_img, aug_boxes = augment_sample(img_mat, sample.bboxes, augmentor)
                    cv2.imwrite(str(aug_img_path), aug_img)

                    aug_lines = [b.to_yolo_line() for b in aug_boxes]
                    aug_lbl_path.write_text("\n".join(aug_lines) + ("\n" if aug_lines else ""), encoding="utf-8")

                    for b in aug_boxes:
                        cname = canonical_names.get(b.class_id, "unknown")
                        class_instances[cname] = class_instances.get(cname, 0) + 1
                else:
                    # Simulation in dry run
                    for b in sample.bboxes:
                        cname = canonical_names.get(b.class_id, "unknown")
                        class_instances[cname] = class_instances.get(cname, 0) + 1

                split_aug_count += 1

        total_processed_images += (split_orig_count + split_aug_count)
        total_augmented_images += split_aug_count

        split_reports[split] = {
            "original_images": split_orig_count,
            "augmented_images": split_aug_count,
            "total_images": split_orig_count + split_aug_count,
            "class_instances": class_instances,
        }

    # Consolidated report
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "processed_dir": str(out_dir),
        "dry_run": dry_run,
        "sources": source_stats or {},
        "split_counts": split_reports,
        "total_processed_images": total_processed_images,
        "augmented_sample_count": total_augmented_images,
        "canonical_classes": {str(k): v for k, v in config.classes.items()},
    }

    if not dry_run:
        report_file = out_dir / "dataset_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Dataset report saved successfully to {report_file}")

    return report


def update_railway_dataset_yaml(
    yaml_path: Union[str, Path],
    processed_dir: Union[str, Path],
    config: Optional[ClassesConfig] = None,
) -> Path:
    """Update or validate backend/configs/railway_dataset.yaml to point to processed splits."""
    if config is None:
        config = load_classes_config()

    yaml_path = Path(yaml_path).resolve()
    processed_dir = Path(processed_dir).resolve()

    # Determine relative path from config directory to processed directory
    try:
        rel_path = os.path.relpath(processed_dir, yaml_path.parent).replace("\\", "/")
    except ValueError:
        rel_path = str(processed_dir).replace("\\", "/")

    content = {
        "path": rel_path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(config.classes),
        "names": {idx: name for idx, name in config.classes.items()},
    }

    header = (
        "# ==============================================================================\n"
        "# AI-Based Multi-Defect Railway Monitoring and Severity Assessment\n"
        "# Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3\n"
        "# Ultralytics YOLO Dataset Configuration (Auto-Generated / Validated)\n"
        "# ==============================================================================\n\n"
    )

    yaml_str = yaml.dump(content, sort_keys=False, default_flow_style=False)
    yaml_path.write_text(header + yaml_str, encoding="utf-8")
    logger.info(f"Updated YOLO dataset configuration at {yaml_path}")
    return yaml_path
