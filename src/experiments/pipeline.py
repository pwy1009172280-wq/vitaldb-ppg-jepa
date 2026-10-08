"""Unique pipeline orchestration glue (authority mapping and execution).

This module owns the single production route from a validated ``PipelineConfig``
to a built model and stage execution. It does not re-implement the training loop
or a second factory; it maps the generic config onto the existing
``src.pretrain.config`` dataclasses, ``build_model``, the adapter, and the
generic ``Trainer``.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.schema import PipelineConfig
from src.data.base import BaseDataset
from src.experiments.metadata import seed_everything
from src.experiments.run import create_run
from src.models.jepa import JEPATrainingAdapter
from src.pretrain.optim import adamw_parameter_groups
from src.training import (
    CheckpointManager,
    CheckpointSelectionPolicy,
    Trainer,
    TrainerConfig,
)


def _subset(section: Mapping[str, Any], cls: type) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in section.items() if key in names}


def build_model_from_config(model_section: Mapping[str, Any]) -> torch.nn.Module:
    """Build a model from the generic ``model`` config section."""
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
    """Rebuild a model from a checkpoint's embedded resolved config."""
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
    """Reconstruct a model from a checkpoint's embedded config."""
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
    adapter = JEPATrainingAdapter(model)
    adapter.load_state_dict(payload["model"])
    return adapter, payload


def _resolved_dict(config: PipelineConfig) -> dict[str, Any]:
    return asdict(config)


def _collate(samples):
    return {
        "waveform": torch.stack(
            [torch.from_numpy(np.asarray(sample.signal, dtype=np.float32)) for sample in samples]
        )
    }


def run_pretrain(
    config: PipelineConfig,
    dataset: BaseDataset,
    *,
    results_root: str | Path,
    run_id: str | None = None,
) -> Path:
    """Run pretraining to a checkpoint; return the checkpoint path.

    Wires the parity invariants: seed-before-init, dedicated mask RNG, parameter
    groups, the real update budget, and device ownership. The dataset is injected
    (fixture or native MIMIC) — this function owns no dataset-specific logic.
    """
    experiment = config.experiment
    seed = int(experiment.get("seed", 0))
    # seed BEFORE any model/optimizer construction (parity)
    seed_everything(seed)

    model = build_model_from_config(config.model)
    adapter = JEPATrainingAdapter(model)
    adapter.set_mask_generator(torch.Generator().manual_seed(seed + 7919))

    device = resolve_device(experiment.get("device"))
    adapter.to(device)

    training = experiment.get("training") or {}
    batch_size = int(training.get("batch_size", 2))
    lr = float(training.get("lr", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.0))
    max_updates = training.get("max_updates")
    epochs = int(training.get("epochs", 1))

    optimizer = torch.optim.AdamW(adamw_parameter_groups(adapter, weight_decay), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    name = run_id or experiment.get("run_id") or f"pretrain-seed-{seed}"
    resolved = _resolved_dict(config)
    run, _ = create_run(
        name, resolved, "manifest://fixture", "split://pretraining-fixture", seed,
        results_root=results_root,
    )
    manager = CheckpointManager(run.checkpoints_path, CheckpointSelectionPolicy("last"))
    trainer = Trainer(
        adapter, optimizer,
        config=TrainerConfig(amp=False, scheduler_step_policy="none"),
        checkpoint_manager=manager,
        protocol_reference="protocol://pretraining/jepa",
        experiment_manifest_reference=str(run.manifest_path),
        resolved_config=resolved,
    )
    trainer.install_signal_handlers()
    trainer.fit(loader, epochs=epochs, max_updates=max_updates)
    return run.checkpoints_path / "last.pt"
