"""Unique pipeline orchestration glue (authority mapping and execution).

This module owns the single production route from a validated ``PipelineConfig``
to a built model and (later) the full stage execution. It does not re-implement
the training loop or a second factory; it maps the generic config onto the
existing ``src.pretrain.config`` dataclasses and ``build_model``.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

import torch


def _subset(section: Mapping[str, Any], cls: type) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in section.items() if key in names}


def build_model_from_config(model_section: Mapping[str, Any]) -> torch.nn.Module:
    """Build a model from the generic ``model`` config section.

    Only the declared ModelConfig/JEPAConfig fields are forwarded; unknown
    nested keys were already rejected by the loader. Legacy data/train defaults
    are never inherited.
    """
    from src.models.factory import build_model
    from src.pretrain.config import (
        Data2VecConfig,
        DataConfig,
        JEPAConfig,
        MAEConfig,
        ModelConfig,
        SSLConfig,
        TrainConfig,
    )

    family = model_section.get("family")
    if family != "jepa":
        raise ValueError(f"unsupported model family for PPG-JEPA: {family!r}")
    model = ModelConfig(**_subset(model_section, ModelConfig))
    jepa = JEPAConfig(**_subset(model_section, JEPAConfig))
    cfg = SSLConfig(
        method="jepa",
        model=model,
        data=DataConfig(),
        train=TrainConfig(),
        mae=MAEConfig(),
        data2vec=Data2VecConfig(),
        jepa=jepa,
    )
    return build_model(cfg)


def reconstruct_model(resolved_config: Mapping[str, Any]) -> torch.nn.Module:
    """Rebuild a model from a checkpoint's embedded resolved config.

    The external YAML/config is never the authority for reconstruction; only the
    resolved model section stored in the checkpoint is consumed.
    """
    model_section = resolved_config.get("model")
    if not isinstance(model_section, Mapping):
        raise ValueError("resolved config has no model section for reconstruction")
    return build_model_from_config(model_section)


def resolve_device(spec: str | None) -> torch.device:
    """Resolve an explicit device spec into a concrete ``torch.device``."""
    if spec in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)
