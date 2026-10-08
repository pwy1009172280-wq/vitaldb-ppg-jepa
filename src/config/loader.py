"""Configuration loading with environment-independent YAML semantics."""

from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import PipelineConfig

_SECTIONS = ("dataset", "preprocessing", "model", "experiment")


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> PipelineConfig:
    """Load a YAML config and optionally replace top-level sections.

    Unknown top-level keys are rejected so configuration drift is visible.
    Deep merge is deliberately deferred until experiment needs require it.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown top-level config sections: {sorted(unknown)}")
    merged = dict(raw)
    if overrides:
        unknown_overrides = set(overrides) - set(_SECTIONS)
        if unknown_overrides:
            raise ValueError(f"unknown override sections: {sorted(unknown_overrides)}")
        merged.update(overrides)
    sections = {name: merged.get(name, {}) for name in _SECTIONS}
    for name, value in sections.items():
        if not isinstance(value, dict):
            raise ValueError(f"config section {name!r} must be a mapping")
    return PipelineConfig(**sections)

