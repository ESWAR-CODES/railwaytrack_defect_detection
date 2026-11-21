"""Configuration loader and schema validator for Railway Defect Detection.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 1: Project Scaffolding & Backend Configuration.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class SeverityLevel(str, Enum):
    """Enumeration of defect severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RatioThresholdConfig(BaseModel):
    """Threshold configuration for continuous ratio metrics (crack, corrosion)."""
    metric: str
    low_max: float
    medium_max: float


class CountThresholdConfig(BaseModel):
    """Threshold configuration for discrete count metrics (fasteners, sleepers)."""
    metric: str
    low_exact: int
    medium_exact: int
    high_min: int


class MandatoryThresholdConfig(BaseModel):
    """Threshold configuration for mandatory severity defects (obstacles)."""
    metric: str
    mandatory_level: SeverityLevel = SeverityLevel.HIGH


class ClassesConfig(BaseModel):
    """Validated configuration for canonical classes, aliases, thresholds, and risk weights."""
    classes: Dict[int, str]
    aliases: Dict[str, str]
    severity_thresholds: Dict[
        str,
        Union[RatioThresholdConfig, CountThresholdConfig, MandatoryThresholdConfig, Dict[str, Any]],
    ]
    risk_weights: Dict[str, int]
    severity_colors: Dict[str, str]

    # Pre-computed reverse lookup for quick access
    _name_to_id: Dict[str, int] = {}
    _normalized_aliases: Dict[str, str] = {}

    @model_validator(mode="after")
    def validate_classes_and_build_lookups(self) -> ClassesConfig:
        """Validate canonical classes and precompute normalized lookup structures."""
        expected_classes = {
            0: "rail_crack",
            1: "missing_fastener",
            2: "rail_corrosion",
            3: "obstacle",
            4: "sleeper_misalignment",
        }

        # Validate exact 5 canonical classes
        for idx, expected_name in expected_classes.items():
            if idx not in self.classes:
                raise ValueError(f"Missing canonical class index {idx} in classes configuration.")
            if self.classes[idx] != expected_name:
                raise ValueError(
                    f"Class index {idx} must be '{expected_name}', got '{self.classes[idx]}'."
                )

        if len(self.classes) != 5:
            raise ValueError(f"Expected exactly 5 canonical classes, got {len(self.classes)}.")

        # Build name to ID index
        name_to_id = {name: idx for idx, name in self.classes.items()}
        object.__setattr__(self, "_name_to_id", name_to_id)

        # Build normalized alias index: lowercase and stripped of special characters
        normalized_aliases: Dict[str, str] = {}

        # Canonical names map to themselves
        for canonical_name in self.classes.values():
            normalized_aliases[canonical_name.lower()] = canonical_name

        # Map all user-provided aliases to their canonical name
        for alias, target in self.aliases.items():
            if target not in name_to_id:
                raise ValueError(
                    f"Alias '{alias}' maps to unknown target class '{target}'."
                )
            clean_alias = alias.strip().lower()
            normalized_aliases[clean_alias] = target
            # Also register space-replaced variant if contains underscore/hyphen
            normalized_aliases[clean_alias.replace("_", " ")] = target
            normalized_aliases[clean_alias.replace("-", " ")] = target
            normalized_aliases[clean_alias.replace(" ", "_")] = target

        object.__setattr__(self, "_normalized_aliases", normalized_aliases)

        # Validate risk weights
        for level in SeverityLevel:
            if level.value not in self.risk_weights and level.name not in self.risk_weights:
                raise ValueError(f"Missing risk weight for severity level {level.value}.")

        # Validate severity colors
        for level in SeverityLevel:
            if level.value not in self.severity_colors and level.name not in self.severity_colors:
                raise ValueError(f"Missing severity color for severity level {level.value}.")

        return self

    def get_class_name(self, class_id: int) -> str:
        """Retrieve canonical class name by numeric class index."""
        if class_id not in self.classes:
            raise KeyError(f"Invalid class ID: {class_id}. Allowed IDs: {list(self.classes.keys())}")
        return self.classes[class_id]

    def get_class_id(self, class_name: str) -> int:
        """Retrieve canonical class index by name or alias."""
        canonical = self.resolve_alias(class_name)
        return self._name_to_id[canonical]

    def resolve_alias(self, raw_label: str) -> str:
        """Resolve a raw label string (e.g. from Roboflow or annotations) to its canonical class name.
        
        Raises ValueError if the label cannot be resolved.
        """
        if not raw_label or not isinstance(raw_label, str):
            raise ValueError(f"Invalid label provided: {raw_label}")

        cleaned = raw_label.strip().lower()
        if cleaned in self._normalized_aliases:
            return self._normalized_aliases[cleaned]

        cleaned_underscore = cleaned.replace(" ", "_").replace("-", "_")
        if cleaned_underscore in self._normalized_aliases:
            return self._normalized_aliases[cleaned_underscore]

        raise ValueError(
            f"Unrecognized label '{raw_label}'. Could not map to any canonical defect class."
        )

    def evaluate_severity(
        self, class_name_or_alias: str, value: Optional[Union[float, int]] = None
    ) -> SeverityLevel:
        """Deterministic severity evaluation based on defect class and metric value.
        
        Rules:
        - rail_crack (diagonal_ratio):
            low: < 0.03
            medium: <= 0.08
            high: > 0.08
        - rail_corrosion (area_ratio):
            low: < 0.05
            medium: <= 0.15
            high: > 0.15
        - missing_fastener (frame_count):
            low: == 1
            medium: == 2
            high: >= 3
        - sleeper_misalignment (frame_count):
            low: == 1
            medium: == 2
            high: >= 3
        - obstacle:
            mandatory: HIGH
        """
        canonical = self.resolve_alias(class_name_or_alias)

        if canonical == "obstacle":
            return SeverityLevel.HIGH

        if value is None:
            raise ValueError(f"Metric value required to evaluate severity for '{canonical}'.")

        if canonical == "rail_crack":
            # diagonal_ratio: low < 0.03, medium <= 0.08, high > 0.08
            ratio = float(value)
            if ratio < 0.03:
                return SeverityLevel.LOW
            elif ratio <= 0.08:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.HIGH

        elif canonical == "rail_corrosion":
            # area_ratio: low < 0.05, medium <= 0.15, high > 0.15
            ratio = float(value)
            if ratio < 0.05:
                return SeverityLevel.LOW
            elif ratio <= 0.15:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.HIGH

        elif canonical in ("missing_fastener", "sleeper_misalignment"):
            # frame_count: low == 1, medium == 2, high >= 3
            count = int(value)
            if count <= 0:
                raise ValueError(f"Count for '{canonical}' must be positive integer, got {count}.")
            if count == 1:
                return SeverityLevel.LOW
            elif count == 2:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.HIGH

        raise ValueError(f"No severity evaluation rules defined for '{canonical}'.")

    def get_risk_weight(self, severity: Union[SeverityLevel, str]) -> int:
        """Return the Track Risk Index weight for a severity level."""
        key = severity.value if isinstance(severity, SeverityLevel) else str(severity).upper()
        if key not in self.risk_weights:
            raise KeyError(f"Unknown severity level '{severity}'. Allowed: {list(self.risk_weights.keys())}")
        return self.risk_weights[key]

    def get_severity_color(self, severity: Union[SeverityLevel, str]) -> str:
        """Return the UI/telemetry hex color code for a severity level."""
        key = severity.value if isinstance(severity, SeverityLevel) else str(severity).upper()
        if key not in self.severity_colors:
            raise KeyError(f"Unknown severity level '{severity}'. Allowed: {list(self.severity_colors.keys())}")
        return self.severity_colors[key]

    def calculate_track_risk_index(
        self, defect_severities: List[Union[SeverityLevel, str]]
    ) -> float:
        """Calculate the composite Track Risk Index (TRI) from a list of defect severities."""
        if not defect_severities:
            return 0.0
        total_weight = sum(self.get_risk_weight(s) for s in defect_severities)
        return float(total_weight)


# Global singleton instance cache
_CONFIG_CACHE: Optional[ClassesConfig] = None


def get_default_config_path() -> Path:
    """Return the absolute path to backend/configs/classes.yaml."""
    return Path(__file__).resolve().parent.parent / "configs" / "classes.yaml"


def load_classes_config(config_path: Optional[Union[str, Path]] = None, reload: bool = False) -> ClassesConfig:
    """Load, validate, and return the ClassesConfig from YAML file."""
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and not reload and config_path is None:
        return _CONFIG_CACHE

    target_path = Path(config_path) if config_path else get_default_config_path()
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = ClassesConfig.model_validate(data)

    if config_path is None or target_path == get_default_config_path():
        _CONFIG_CACHE = config

    return config


def resolve_alias(raw_label: str) -> str:
    """Convenience helper to resolve a raw label using the default loaded configuration."""
    return load_classes_config().resolve_alias(raw_label)


def evaluate_severity(class_name: str, value: Optional[Union[float, int]] = None) -> SeverityLevel:
    """Convenience helper to evaluate severity using the default loaded configuration."""
    return load_classes_config().evaluate_severity(class_name, value)
