"""Unit tests for Railway Defect Monitoring configuration loader, alias resolution, and severity rules.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 1: Verification Gate.
"""

from pathlib import Path
import pytest
import yaml

from backend.engine.config_loader import (
    ClassesConfig,
    SeverityLevel,
    evaluate_severity,
    get_default_config_path,
    load_classes_config,
    resolve_alias,
)


@pytest.fixture
def config() -> ClassesConfig:
    """Fixture providing freshly loaded ClassesConfig."""
    return load_classes_config(reload=True)


class TestCanonicalClasses:
    """Tests verifying the exact 5 canonical defect classes."""

    def test_canonical_class_count_and_mapping(self, config: ClassesConfig):
        """Verify exactly 5 canonical classes with exact index mappings."""
        assert len(config.classes) == 5
        assert config.classes[0] == "rail_crack"
        assert config.classes[1] == "missing_fastener"
        assert config.classes[2] == "rail_corrosion"
        assert config.classes[3] == "obstacle"
        assert config.classes[4] == "sleeper_misalignment"

    def test_get_class_name(self, config: ClassesConfig):
        """Verify class name lookup by integer ID."""
        assert config.get_class_name(0) == "rail_crack"
        assert config.get_class_name(1) == "missing_fastener"
        assert config.get_class_name(2) == "rail_corrosion"
        assert config.get_class_name(3) == "obstacle"
        assert config.get_class_name(4) == "sleeper_misalignment"

        with pytest.raises(KeyError):
            config.get_class_name(5)

    def test_get_class_id(self, config: ClassesConfig):
        """Verify class ID lookup by canonical name or alias."""
        assert config.get_class_id("rail_crack") == 0
        assert config.get_class_id("missing_fastener") == 1
        assert config.get_class_id("rail_corrosion") == 2
        assert config.get_class_id("obstacle") == 3
        assert config.get_class_id("sleeper_misalignment") == 4

        # Using aliases
        assert config.get_class_id("crack") == 0
        assert config.get_class_id("fastener") == 1
        assert config.get_class_id("rust") == 2
        assert config.get_class_id("rock") == 3
        assert config.get_class_id("misaligned_sleeper") == 4


class TestAliasResolution:
    """Tests verifying robust label normalization and Roboflow variant alias resolution."""

    @pytest.mark.parametrize(
        "raw_label,expected_canonical",
        [
            # rail_crack aliases
            ("crack", "rail_crack"),
            ("broken_rail", "rail_crack"),
            ("broken_rail_low", "rail_crack"),
            ("broken_rail_medium", "rail_crack"),
            ("broken_rail_high", "rail_crack"),
            ("glue_crack", "rail_crack"),
            ("head_checks", "rail_crack"),
            ("head_checks_low", "rail_crack"),
            ("head_checks_medium", "rail_crack"),
            ("head_checks_high", "rail_crack"),
            ("squats", "rail_crack"),
            ("squats_low", "rail_crack"),
            ("squats_medium", "rail_crack"),
            ("squats_high", "rail_crack"),
            ("railway_join_bar_defects", "rail_crack"),
            ("railway_join_bar_defects_low", "rail_crack"),
            ("railway_join_bar_defects_medium", "rail_crack"),
            ("railway_join_bar_defects_high", "rail_crack"),
            ("transverse_crack", "rail_crack"),
            ("longitudinal_crack", "rail_crack"),
            ("rail_crack", "rail_crack"),
            # missing_fastener aliases
            ("fastener", "missing_fastener"),
            ("broken_fastener", "missing_fastener"),
            ("missing_clip", "missing_fastener"),
            ("damaged_fastener", "missing_fastener"),
            ("missing_bolt", "missing_fastener"),
            ("loose_fastener", "missing_fastener"),
            ("clip_missing", "missing_fastener"),
            ("missing_fastener", "missing_fastener"),
            # rail_corrosion aliases
            ("corrosion", "rail_corrosion"),
            ("corrosion_low", "rail_corrosion"),
            ("corrosion_medium", "rail_corrosion"),
            ("corrosion_high", "rail_corrosion"),
            ("rust", "rail_corrosion"),
            ("rusting", "rail_corrosion"),
            ("surface_corrosion", "rail_corrosion"),
            ("rail_corrosion", "rail_corrosion"),
            # obstacle aliases
            ("object", "obstacle"),
            ("foreign_object", "obstacle"),
            ("rock", "obstacle"),
            ("stone", "obstacle"),
            ("debris", "obstacle"),
            ("boulder", "obstacle"),
            ("obstacle", "obstacle"),
            # sleeper_misalignment aliases
            ("sleeper", "sleeper_misalignment"),
            ("misaligned_sleeper", "sleeper_misalignment"),
            ("broken_sleeper", "sleeper_misalignment"),
            ("tie_defect", "sleeper_misalignment"),
            ("tie_misalignment", "sleeper_misalignment"),
            ("sleeper_defect", "sleeper_misalignment"),
            ("sleeper_misalignment", "sleeper_misalignment"),
        ],
    )
    def test_alias_mapping(self, config: ClassesConfig, raw_label: str, expected_canonical: str):
        """Verify each expected Roboflow/raw label maps to its canonical class."""
        assert config.resolve_alias(raw_label) == expected_canonical

    def test_case_and_whitespace_insensitivity(self, config: ClassesConfig):
        """Verify alias resolution handles uppercase, mixed case, and extra whitespace."""
        assert config.resolve_alias("  CRACK  ") == "rail_crack"
        assert config.resolve_alias("Broken_Rail") == "rail_crack"
        assert config.resolve_alias("MISSING_FASTENER") == "missing_fastener"
        assert config.resolve_alias("broken fastener") == "missing_fastener"
        assert config.resolve_alias("Rust") == "rail_corrosion"
        assert config.resolve_alias("FOREIGN-OBJECT") == "obstacle"
        assert config.resolve_alias("Misaligned Sleeper") == "sleeper_misalignment"

    def test_module_level_resolve_alias_helper(self):
        """Verify standalone resolve_alias function."""
        assert resolve_alias("crack") == "rail_crack"
        assert resolve_alias("rust") == "rail_corrosion"

    def test_unrecognized_label_raises_error(self, config: ClassesConfig):
        """Verify unrecognized labels raise a descriptive ValueError."""
        with pytest.raises(ValueError, match="Unrecognized label"):
            config.resolve_alias("unknown_defect_xyz")

        with pytest.raises(ValueError, match="Invalid label"):
            config.resolve_alias("")


class TestSeverityThresholdRules:
    """Tests verifying deterministic severity classification per class."""

    def test_rail_crack_thresholds(self, config: ClassesConfig):
        """rail_crack: diagonal_ratio low < 0.03, medium <= 0.08, high > 0.08."""
        # Low: diagonal_ratio < 0.03
        assert config.evaluate_severity("rail_crack", 0.01) == SeverityLevel.LOW
        assert config.evaluate_severity("rail_crack", 0.029) == SeverityLevel.LOW

        # Medium: 0.03 <= diagonal_ratio <= 0.08
        assert config.evaluate_severity("rail_crack", 0.03) == SeverityLevel.MEDIUM
        assert config.evaluate_severity("rail_crack", 0.05) == SeverityLevel.MEDIUM
        assert config.evaluate_severity("rail_crack", 0.08) == SeverityLevel.MEDIUM

        # High: diagonal_ratio > 0.08
        assert config.evaluate_severity("rail_crack", 0.081) == SeverityLevel.HIGH
        assert config.evaluate_severity("rail_crack", 0.15) == SeverityLevel.HIGH

        # Via alias
        assert config.evaluate_severity("crack", 0.02) == SeverityLevel.LOW
        assert config.evaluate_severity("broken_rail", 0.09) == SeverityLevel.HIGH

    def test_rail_corrosion_thresholds(self, config: ClassesConfig):
        """rail_corrosion: area_ratio low < 0.05, medium <= 0.15, high > 0.15."""
        # Low: area_ratio < 0.05
        assert config.evaluate_severity("rail_corrosion", 0.01) == SeverityLevel.LOW
        assert config.evaluate_severity("rail_corrosion", 0.049) == SeverityLevel.LOW

        # Medium: 0.05 <= area_ratio <= 0.15
        assert config.evaluate_severity("rail_corrosion", 0.05) == SeverityLevel.MEDIUM
        assert config.evaluate_severity("rail_corrosion", 0.10) == SeverityLevel.MEDIUM
        assert config.evaluate_severity("rail_corrosion", 0.15) == SeverityLevel.MEDIUM

        # High: area_ratio > 0.15
        assert config.evaluate_severity("rail_corrosion", 0.151) == SeverityLevel.HIGH
        assert config.evaluate_severity("rail_corrosion", 0.30) == SeverityLevel.HIGH

        # Via alias
        assert config.evaluate_severity("rust", 0.04) == SeverityLevel.LOW
        assert config.evaluate_severity("corrosion", 0.12) == SeverityLevel.MEDIUM

    def test_missing_fastener_thresholds(self, config: ClassesConfig):
        """missing_fastener: frame_count low == 1, medium == 2, high >= 3."""
        assert config.evaluate_severity("missing_fastener", 1) == SeverityLevel.LOW
        assert config.evaluate_severity("missing_fastener", 2) == SeverityLevel.MEDIUM
        assert config.evaluate_severity("missing_fastener", 3) == SeverityLevel.HIGH
        assert config.evaluate_severity("missing_fastener", 5) == SeverityLevel.HIGH

        # Via alias
        assert config.evaluate_severity("broken_fastener", 1) == SeverityLevel.LOW
        assert config.evaluate_severity("missing_clip", 4) == SeverityLevel.HIGH

    def test_sleeper_misalignment_thresholds(self, config: ClassesConfig):
        """sleeper_misalignment: frame_count low == 1, medium == 2, high >= 3."""
        assert config.evaluate_severity("sleeper_misalignment", 1) == SeverityLevel.LOW
        assert config.evaluate_severity("sleeper_misalignment", 2) == SeverityLevel.MEDIUM
        assert config.evaluate_severity("sleeper_misalignment", 3) == SeverityLevel.HIGH
        assert config.evaluate_severity("sleeper_misalignment", 6) == SeverityLevel.HIGH

        # Via alias
        assert config.evaluate_severity("misaligned_sleeper", 1) == SeverityLevel.LOW
        assert config.evaluate_severity("sleeper", 3) == SeverityLevel.HIGH

    def test_obstacle_thresholds(self, config: ClassesConfig):
        """obstacle: mandatory high."""
        assert config.evaluate_severity("obstacle") == SeverityLevel.HIGH
        assert config.evaluate_severity("obstacle", value=10) == SeverityLevel.HIGH
        assert config.evaluate_severity("rock") == SeverityLevel.HIGH
        assert config.evaluate_severity("foreign_object", None) == SeverityLevel.HIGH

    def test_module_level_evaluate_severity_helper(self):
        """Verify standalone evaluate_severity function."""
        assert evaluate_severity("crack", 0.02) == SeverityLevel.LOW
        assert evaluate_severity("obstacle") == SeverityLevel.HIGH


class TestTrackRiskIndexAndColors:
    """Tests verifying risk weights, hex colors, and composite score computation."""

    def test_risk_weights(self, config: ClassesConfig):
        """LOW: 10, MEDIUM: 25, HIGH: 50."""
        assert config.get_risk_weight(SeverityLevel.LOW) == 10
        assert config.get_risk_weight(SeverityLevel.MEDIUM) == 25
        assert config.get_risk_weight(SeverityLevel.HIGH) == 50

        assert config.get_risk_weight("low") == 10
        assert config.get_risk_weight("medium") == 25
        assert config.get_risk_weight("high") == 50

    def test_severity_colors(self, config: ClassesConfig):
        """LOW: '#D97706', MEDIUM: '#EA580C', HIGH: '#DC2626'."""
        assert config.get_severity_color(SeverityLevel.LOW) == "#D97706"
        assert config.get_severity_color(SeverityLevel.MEDIUM) == "#EA580C"
        assert config.get_severity_color(SeverityLevel.HIGH) == "#DC2626"

        assert config.get_severity_color("low") == "#D97706"
        assert config.get_severity_color("medium") == "#EA580C"
        assert config.get_severity_color("high") == "#DC2626"

    def test_track_risk_index_calculation(self, config: ClassesConfig):
        """Verify composite TRI calculation for empty and multi-defect scenarios."""
        assert config.calculate_track_risk_index([]) == 0.0

        # 1 LOW (10) + 1 MEDIUM (25) + 1 HIGH (50) = 85.0
        severities = [SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH]
        assert config.calculate_track_risk_index(severities) == 85.0

        # String inputs
        str_severities = ["LOW", "HIGH"]
        assert config.calculate_track_risk_index(str_severities) == 60.0


class TestYoloDatasetConfig:
    """Tests verifying backend/configs/railway_dataset.yaml adheres to Ultralytics YOLO specs."""

    def test_railway_dataset_yaml(self):
        """Verify path, image splits, nc, and names match canonical classes."""
        dataset_yaml_path = (
            Path(__file__).resolve().parent.parent / "configs" / "railway_dataset.yaml"
        )
        assert dataset_yaml_path.exists(), f"Missing dataset yaml at {dataset_yaml_path}"

        with open(dataset_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["path"] == "../data/processed"
        assert data["train"] == "images/train"
        assert data["val"] == "images/val"
        assert data["test"] == "images/test"
        assert data["nc"] == 5

        names = data["names"]
        assert names[0] == "rail_crack"
        assert names[1] == "missing_fastener"
        assert names[2] == "rail_corrosion"
        assert names[3] == "obstacle"
        assert names[4] == "sleeper_misalignment"
