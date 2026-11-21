"""Unit and Integration Tests for Railway Defect Dataset Harmonization, Merging, Splitting & Drone Augmentation.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 2: Test Suite.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import yaml

from backend.engine.config_loader import ClassesConfig, load_classes_config
from backend.engine.dataset_pipeline import (
    BBox,
    SampleItem,
    augment_sample,
    clamp_and_validate_bbox,
    create_drone_augmentation_pipeline,
    find_dataset_root,
    find_image_label_pairs,
    load_and_harmonize_dataset,
    make_unique_stem,
    parse_dataset_classes,
    split_dataset,
    update_railway_dataset_yaml,
    write_processed_dataset,
)


@pytest.fixture
def config() -> ClassesConfig:
    """Fixture providing freshly loaded ClassesConfig."""
    return load_classes_config()


class TestBBoxClampingAndValidation:
    """Tests verifying bounding box coordinate clamping and degenerate box filtering."""

    def test_valid_bbox_coordinates(self):
        """A normal bounding box within [0.0, 1.0] remains valid."""
        box = clamp_and_validate_bbox(class_id=0, x_c=0.5, y_c=0.5, w=0.2, h=0.3)
        assert box is not None
        assert box.class_id == 0
        assert box.x_center == 0.5
        assert box.y_center == 0.5
        assert box.width == 0.2
        assert box.height == 0.3

    def test_out_of_bounds_clamping(self):
        """Coordinates exceeding 1.0 or below 0.0 are clamped cleanly."""
        # Box extends past right edge: center 0.95, width 0.2 -> right edge 1.05
        box = clamp_and_validate_bbox(class_id=1, x_c=0.95, y_c=0.5, w=0.2, h=0.2)
        assert box is not None
        # Right edge must be clamped to 1.0, left edge is 0.85 -> new center 0.925, width 0.15
        assert box.x_center + box.width / 2.0 <= 1.000001
        assert box.x_center - box.width / 2.0 >= 0.0

        # Box extends past top-left: center 0.05, width 0.2 -> left edge -0.05
        box_left = clamp_and_validate_bbox(class_id=2, x_c=0.05, y_c=0.05, w=0.2, h=0.2)
        assert box_left is not None
        assert box_left.x_center - box_left.width / 2.0 >= -1e-6
        assert box_left.y_center - box_left.height / 2.0 >= -1e-6

    def test_degenerate_and_invalid_inputs(self):
        """Degenerate, negative, zero-size, or NaN values return None."""
        assert clamp_and_validate_bbox(0, 0.5, 0.5, 0.0, 0.2) is None
        assert clamp_and_validate_bbox(0, 0.5, 0.5, -0.1, 0.2) is None
        assert clamp_and_validate_bbox(0, float("nan"), 0.5, 0.2, 0.2) is None
        assert clamp_and_validate_bbox(0, 0.5, float("inf"), 0.2, 0.2) is None
        # Tiny degenerate width
        assert clamp_and_validate_bbox(0, 0.5, 0.5, 1e-6, 1e-6) is None

    def test_to_yolo_line_and_albumentations(self):
        """Verify format conversion methods."""
        box = BBox(class_id=3, x_center=0.456789, y_center=0.567891, width=0.123456, height=0.234567)
        line = box.to_yolo_line()
        parts = line.split()
        assert len(parts) == 5
        assert parts[0] == "3"
        assert float(parts[1]) == pytest.approx(0.456789, rel=1e-4)

        alb_box = box.to_albumentations_bbox()
        assert len(alb_box) == 4
        assert alb_box == pytest.approx([0.456789, 0.567891, 0.123456, 0.234567], rel=1e-5)


class TestDatasetDiscoveryAndClassParsing:
    """Tests verifying robust nested folder detection and data.yaml parsing."""

    def test_find_dataset_root_direct(self, tmp_path: Path):
        """Direct directory with data.yaml is resolved immediately."""
        (tmp_path / "data.yaml").write_text("names: [crack]", encoding="utf-8")
        assert find_dataset_root(tmp_path) == tmp_path

    def test_find_dataset_root_nested_zip(self, tmp_path: Path):
        """Nested subfolder from unzipped archive is detected automatically."""
        nested = tmp_path / "unzipped_subfolder" / "inner_dataset"
        nested.mkdir(parents=True)
        (nested / "data.yaml").write_text("names: [crack]", encoding="utf-8")
        assert find_dataset_root(tmp_path) == nested

    def test_parse_dataset_classes_list_and_dict(self, tmp_path: Path):
        """Supports both YAML list format and dict format for names."""
        list_yaml = tmp_path / "list_data.yaml"
        list_yaml.write_text("names: ['crack', 'fastener', 'corrosion']", encoding="utf-8")
        classes_list = parse_dataset_classes(list_yaml)
        assert classes_list == {0: "crack", 1: "fastener", 2: "corrosion"}

        dict_yaml = tmp_path / "dict_data.yaml"
        dict_yaml.write_text("names:\n  0: crack\n  1: fastener", encoding="utf-8")
        classes_dict = parse_dataset_classes(dict_yaml)
        assert classes_dict == {0: "crack", 1: "fastener"}

        assert parse_dataset_classes(tmp_path / "non_existent.yaml") == {}


class TestImageLabelPairing:
    """Tests verifying pair discovery between image files and YOLO text labels."""

    def test_standard_yolo_structure_pairing(self, tmp_path: Path):
        """Standard split/images and split/labels directories are paired."""
        img_dir = tmp_path / "train" / "images"
        lbl_dir = tmp_path / "train" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        img_file = img_dir / "sample_01.jpg"
        img_file.write_bytes(b"\x00" * 10)
        lbl_file = lbl_dir / "sample_01.txt"
        lbl_file.write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")

        pairs = find_image_label_pairs(tmp_path)
        assert len(pairs) == 1
        found_img, found_lbl = pairs[0]
        assert found_img == img_file
        assert found_lbl == lbl_file

    def test_background_image_without_label(self, tmp_path: Path):
        """An image with no label file is discovered with label=None."""
        img_dir = tmp_path / "test" / "images"
        img_dir.mkdir(parents=True)
        img_file = img_dir / "bg_image.png"
        img_file.write_bytes(b"\x00" * 10)

        pairs = find_image_label_pairs(tmp_path)
        assert len(pairs) == 1
        found_img, found_lbl = pairs[0]
        assert found_img == img_file
        assert found_lbl is None


class TestHarmonizationAndAliasMapping:
    """Tests verifying Roboflow alias remapping, unmapped dropping, and line skipping."""

    def test_load_and_harmonize_synthetic_dataset(self, tmp_path: Path, config: ClassesConfig):
        """Synthetic dataset with known aliases, unmapped classes, and malformed lines."""
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            "names: ['crack', 'fastener_present', 'broken_fastener', 'rust', 'rock', 'misaligned_sleeper']",
            encoding="utf-8",
        )

        img_dir = tmp_path / "train" / "images"
        lbl_dir = tmp_path / "train" / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Image 1: Contains crack (0 -> rail_crack), fastener_present (1 -> unmapped/drop), broken_fastener (2 -> missing_fastener)
        im1 = img_dir / "test1.jpg"
        im1.write_bytes(b"\x00" * 10)
        (lbl_dir / "test1.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n"  # crack -> 0
            "1 0.3 0.3 0.1 0.1\n"  # fastener_present -> unmapped, should drop
            "2 0.7 0.7 0.15 0.15\n",  # broken_fastener -> 1
            encoding="utf-8",
        )

        # Image 2: Contains rust (3 -> rail_corrosion), rock (4 -> obstacle), sleeper (5 -> sleeper_misalignment), malformed line
        im2 = img_dir / "test2.jpg"
        im2.write_bytes(b"\x00" * 10)
        (lbl_dir / "test2.txt").write_text(
            "3 0.4 0.4 0.2 0.2\n"  # rust -> 2
            "4 0.8 0.8 0.1 0.1\n"  # rock -> 3
            "5 0.2 0.8 0.3 0.1\n"  # misaligned_sleeper -> 4
            "invalid malformed line\n"
            "0 0.5\n",  # incomplete line
            encoding="utf-8",
        )

        samples, stats = load_and_harmonize_dataset(tmp_path, source_tag="test_src", config=config)

        assert len(samples) == 2
        assert stats["total_boxes_loaded"] == 5  # 2 in im1 (1 dropped) + 3 in im2
        assert stats["malformed_lines_skipped"] == 2
        assert stats["dropped_unmapped_classes"].get("fastener_present") == 1

        # Check sample 1 bounding box classes
        sample1 = next(s for s in samples if s.image_path.name == "test1.jpg")
        classes_s1 = [b.class_id for b in sample1.bboxes]
        assert classes_s1 == [0, 1]  # rail_crack (0) and missing_fastener (1)

        # Check sample 2 bounding box classes
        sample2 = next(s for s in samples if s.image_path.name == "test2.jpg")
        classes_s2 = [b.class_id for b in sample2.bboxes]
        assert classes_s2 == [2, 3, 4]  # rail_corrosion, obstacle, sleeper_misalignment


class TestDeterministicSplittingAndDeduplication:
    """Tests verifying 70/20/10 split proportions, reproducibility, and collision avoidance."""

    def test_70_20_10_split_proportions(self):
        """100 samples are split into 70 train, 20 val, and 10 test."""
        synthetic_samples = [
            SampleItem(image_path=Path(f"img_{i}.jpg"), source_tag="dsa")
            for i in range(100)
        ]
        splits = split_dataset(synthetic_samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42)

        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 20
        assert len(splits["test"]) == 10

    def test_seed_reproducibility(self):
        """Identical seed produces identical splits; different seed differs."""
        samples = [SampleItem(image_path=Path(f"img_{i}.jpg"), source_tag="dsa") for i in range(50)]
        split_a = split_dataset(samples, seed=42)
        split_b = split_dataset(samples, seed=42)
        split_c = split_dataset(samples, seed=99)

        assert [s.image_path.name for s in split_a["train"]] == [s.image_path.name for s in split_b["train"]]
        assert [s.image_path.name for s in split_a["train"]] != [s.image_path.name for s in split_c["train"]]

    def test_make_unique_stem_deduplication(self):
        """Prefixing and deduplication prevent filename collisions."""
        seen = set()
        stem1 = make_unique_stem(seen, prefix="dsa", original_stem="frame_001")
        stem2 = make_unique_stem(seen, prefix="dsb", original_stem="frame_001")
        stem3 = make_unique_stem(seen, prefix="dsa", original_stem="frame_001")

        assert stem1 == "dsa_frame_001"
        assert stem2 == "dsb_frame_001"
        assert stem3 == "dsa_frame_001_1"


class TestDroneVisionAugmentation:
    """Tests verifying Albumentations drone perturbations and bbox validity."""

    def test_drone_augmentation_pipeline(self):
        """Augmentation produces valid images and coordinates within [0.0, 1.0]."""
        augmentor = create_drone_augmentation_pipeline()

        # Create a synthetic 100x100 BGR image
        image = np.full((100, 100, 3), 120, dtype=np.uint8)
        # Create 2 bounding boxes
        boxes = [
            BBox(class_id=0, x_center=0.4, y_center=0.4, width=0.2, height=0.2),
            BBox(class_id=1, x_center=0.7, y_center=0.7, width=0.15, height=0.15),
        ]

        aug_image, aug_boxes = augment_sample(image, boxes, augmentor)

        assert aug_image.shape == (100, 100, 3)
        assert isinstance(aug_boxes, list)
        for b in aug_boxes:
            assert 0.0 <= b.x_center <= 1.0
            assert 0.0 <= b.y_center <= 1.0
            assert 0.0 < b.width <= 1.0
            assert 0.0 < b.height <= 1.0
            assert b.class_id in (0, 1)

    def test_empty_bboxes_augmentation(self):
        """Augmenting a background image with no bboxes succeeds without error."""
        augmentor = create_drone_augmentation_pipeline()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        aug_image, aug_boxes = augment_sample(image, [], augmentor)
        assert aug_image.shape == (100, 100, 3)
        assert aug_boxes == []


class TestEndToEndDatasetWritingAndReport:
    """Tests verifying file output structure, report generation, and dataset YAML updates."""

    def test_write_processed_dataset_and_report(self, tmp_path: Path, config: ClassesConfig):
        """Writes processed images and labels, verifying directory structure and dataset_report.json."""
        out_dir = tmp_path / "processed"

        # Create a real synthetic image file on disk
        img_file = tmp_path / "input.jpg"
        dummy_img = np.full((80, 80, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(img_file), dummy_img)

        sample = SampleItem(
            image_path=img_file,
            source_tag="dsa",
            bboxes=[BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2)],
        )

        splits = {
            "train": [sample],
            "val": [sample],
            "test": [sample],
        }

        report = write_processed_dataset(
            splits=splits,
            out_dir=out_dir,
            config=config,
            augment_factor=1.0,  # 1 augmented image in train
            seed=42,
            dry_run=False,
        )

        # Check directory structure
        assert (out_dir / "images" / "train").exists()
        assert (out_dir / "images" / "val").exists()
        assert (out_dir / "images" / "test").exists()
        assert (out_dir / "labels" / "train").exists()
        assert (out_dir / "labels" / "val").exists()
        assert (out_dir / "labels" / "test").exists()

        # Check report
        report_path = out_dir / "dataset_report.json"
        assert report_path.exists()
        loaded_report = json.loads(report_path.read_text(encoding="utf-8"))
        assert loaded_report["total_processed_images"] == 4  # 3 original + 1 aug in train
        assert loaded_report["canonical_classes"]["0"] == "rail_crack"

    def test_dry_run_mode(self, tmp_path: Path, config: ClassesConfig):
        """Dry-run returns complete report without creating folders on disk."""
        out_dir = tmp_path / "dry_run_processed"
        sample = SampleItem(image_path=Path("fake.jpg"), source_tag="dsa")
        splits = {"train": [sample], "val": [], "test": []}

        report = write_processed_dataset(
            splits=splits,
            out_dir=out_dir,
            config=config,
            dry_run=True,
        )

        assert report["dry_run"] is True
        assert not out_dir.exists()

    def test_update_railway_dataset_yaml(self, tmp_path: Path, config: ClassesConfig):
        """Updates railway_dataset.yaml with processed directory."""
        yaml_file = tmp_path / "railway_dataset.yaml"
        yaml_file.write_text("old_content: true", encoding="utf-8")

        processed_dir = tmp_path / "processed"
        update_railway_dataset_yaml(yaml_file, processed_dir, config=config)

        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["nc"] == 5
        assert data["train"] == "images/train"
        assert data["val"] == "images/val"
        assert data["test"] == "images/test"
        assert data["names"][0] == "rail_crack"
        assert data["names"][4] == "sleeper_misalignment"
