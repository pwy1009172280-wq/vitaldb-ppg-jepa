"""Configuration loading with nested validation and a PPG pretrain role guard.

Unknown top-level and nested keys are rejected so configuration drift is
visible. Scientific values that remain undecided must be marked
``PI_DECISION_REQUIRED`` and are rejected at load time. The project-level role
guard refuses VitalDB/ECG datasets as PPG-JEPA pretraining inputs; it does not
rewrite the shared Data Bank role policy.
"""

from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import (
    FIXTURE_NAMESPACE_PREFIXES,
    FORBIDDEN_PRETRAIN_DATASETS,
    PI_DECISION_REQUIRED,
    PPG_PRETRAIN_DATASET,
    PPG_PRETRAIN_MODALITY,
    PipelineConfig,
)

_SECTIONS = ("dataset", "preprocessing", "model", "experiment")

# Bounded single-level nested key schema. Extend explicitly as the pipeline
# consumes new fields; do not silently fall back to legacy/orphan shapes.
_ALLOWED_NESTED: dict[str, frozenset[str]] = {
    "dataset": frozenset({"pretrain", "downstream", "role_policy_ref"}),
    "preprocessing": frozenset({"pretrain", "downstream"}),
    "model": frozenset({
        "family",
        # ModelConfig (flat; reused by src.pretrain.config.ModelConfig)
        "patch_size", "patch_stride", "embed_dim", "depth", "num_heads", "mlp_ratio",
        "dropout", "decoder_dim", "decoder_depth", "decoder_num_heads", "decoder_mlp_ratio",
        # JEPAConfig
        "num_target_blocks", "target_block_length", "predictor_dim", "predictor_depth",
        "predictor_num_heads", "predictor_mlp_ratio", "ema_momentum", "loss_beta",
    }),
    "experiment": frozenset({
        "mode", "smoke_only", "seed", "run_id", "representations", "pooling",
        "training", "checkpoint", "task", "protocol", "split", "metrics",
        "aggregation", "selection", "capacity_limits", "device", "budget",
    }),
}


def _reject_pi_placeholders(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_pi_placeholders(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_pi_placeholders(item, f"{path}[{index}]")
    elif value == PI_DECISION_REQUIRED:
        raise ValueError(f"unresolved PI decision at {path}: {PI_DECISION_REQUIRED}")


def _validate_nested(sections: Mapping[str, Any]) -> None:
    for section, allowed in _ALLOWED_NESTED.items():
        value = sections.get(section)
        if not value:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"config section {section!r} must be a mapping")
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown nested keys in {section!r}: {sorted(unknown)}")


def _validate_pretrain_role(sections: Mapping[str, Any]) -> None:
    dataset = sections.get("dataset") or {}
    pretrain = dataset.get("pretrain")
    if not pretrain:
        return
    if not isinstance(pretrain, Mapping):
        raise ValueError("dataset.pretrain must be a mapping")
    name = pretrain.get("name")
    if not name:
        raise ValueError("dataset.pretrain.name is required")
    if name in FORBIDDEN_PRETRAIN_DATASETS:
        raise ValueError(f"dataset {name!r} is forbidden for PPG-JEPA pretraining")
    modality = pretrain.get("modality")
    if modality is not None and str(modality).upper() != PPG_PRETRAIN_MODALITY:
        raise ValueError(f"pretrain modality must be {PPG_PRETRAIN_MODALITY!r}, got {modality!r}")
    if name != PPG_PRETRAIN_DATASET:
        if not name.startswith(FIXTURE_NAMESPACE_PREFIXES):
            raise ValueError(f"pretrain dataset {name!r} is not an approved PPG pretraining source")
        experiment = sections.get("experiment") or {}
        if not experiment.get("smoke_only"):
            raise ValueError("fixture pretrain dataset requires experiment.smoke_only=true")


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> PipelineConfig:
    """Load a YAML config, validate it, and return a ``PipelineConfig``."""
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
    sections: dict[str, Any] = {name: merged.get(name, {}) for name in _SECTIONS}
    for name, value in sections.items():
        if value is None:
            sections[name] = {}
        elif not isinstance(value, dict):
            raise ValueError(f"config section {name!r} must be a mapping")
    _validate_nested(sections)
    _reject_pi_placeholders(sections, "config")
    _validate_pretrain_role(sections)
    return PipelineConfig(**sections)
