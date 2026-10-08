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


def load_checkpoint_model(
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
    expected_model_section: Mapping[str, Any] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct a frozen-able model from a checkpoint's embedded config.

    The external YAML/config is never the reconstruction authority: if
    ``expected_model_section`` is supplied it must content-match the embedded
    model section, otherwise reconstruction fails even when weights would load.
    """
    from src.provenance import content_hash

    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if payload.get("checkpoint_format") != "generic_v1":
        raise ValueError("reconstruction requires a generic_v1 checkpoint")
    model_section = (payload.get("resolved_config") or {}).get("model")
    if not isinstance(model_section, Mapping):
        raise ValueError("checkpoint has no embedded model config for reconstruction")
    if expected_model_section is not None:
        if content_hash("model", expected_model_section) != content_hash("model", model_section):
            raise ValueError("external model config does not match checkpoint")
    model = build_model_from_config(model_section)
    model.load_state_dict(payload["model"])
    return model, payload
