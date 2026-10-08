"""Explicit checkpoint persistence and restoration."""

from dataclasses import dataclass
import os
import random
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from src.experiments.metadata import serialize_config


def _canonical_config(config: Any) -> Any:
    if config is None:
        return None
    return serialize_config(config)


@dataclass(frozen=True)
class CheckpointSelectionPolicy:
    kind: str = "last"
    monitor: str | None = None
    mode: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("last", "monitored_metric"):
            raise ValueError("checkpoint selection kind must be last or monitored_metric")
        if self.kind == "monitored_metric" and (not self.monitor or self.mode not in ("min", "max")):
            raise ValueError("monitored_metric requires monitor and mode=min/max")


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    def __init__(self, directory: str | Path, policy: CheckpointSelectionPolicy) -> None:
        self.directory = Path(directory)
        self.policy = policy
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.directory / f"{name}.pt"

    def save(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        *,
        epoch: int,
        global_step: int,
        best_metric_name: str | None,
        best_metric_value: float | None,
        experiment_manifest_reference: str | None,
        evaluation_protocol_reference: str,
        resolved_config: Any,
        epoch_complete: bool = False,
        next_batch_idx: int = 0,
    ) -> Path:
        payload = {
            "checkpoint_format": "generic_v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "amp_scaler": scaler.state_dict() if scaler is not None else None,
            "rng": _rng_state(),
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "next_batch_idx": next_batch_idx,
            "global_step": global_step,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "experiment_manifest_reference": experiment_manifest_reference,
            "evaluation_protocol_reference": evaluation_protocol_reference,
            "resolved_config": _canonical_config(resolved_config),
            "checkpoint_selection_policy": {
                "kind": self.policy.kind,
                "monitor": self.policy.monitor,
                "mode": self.policy.mode,
            },
        }
        path = self._path(name)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=f".{name}.", suffix=".tmp", delete=False) as handle:
                temporary_path = Path(handle.name)
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return path

    def save_best(self, *args: Any, **kwargs: Any) -> Path:
        """Save the explicitly selected monitored-metric checkpoint."""
        if self.policy.kind != "monitored_metric":
            raise ValueError("save_best requires monitored_metric policy")
        return self.save("best", *args, **kwargs)

    def load(
        self,
        selection: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Any,
        *,
        map_location: str | torch.device = "cpu",
        expected_resolved_config: Any = None,
        expected_experiment_manifest_reference: str | None = None,
        expected_evaluation_protocol_reference: str | None = None,
    ) -> dict[str, Any]:
        if selection not in ("last", "best"):
            raise ValueError("checkpoint selection must be last or best")
        if selection == "best" and self.policy.kind != "monitored_metric":
            raise ValueError("best selection requires monitored_metric policy")
        payload = torch.load(self._path(selection), map_location=map_location, weights_only=False)
        expected_policy = {
            "kind": self.policy.kind,
            "monitor": self.policy.monitor,
            "mode": self.policy.mode,
        }
        if payload.get("checkpoint_selection_policy") != expected_policy:
            raise ValueError("checkpoint selection policy does not match current run")
        if payload.get("resolved_config") != _canonical_config(expected_resolved_config):
            raise ValueError("checkpoint resolved config does not match current run")
        if payload.get("experiment_manifest_reference") != expected_experiment_manifest_reference:
            raise ValueError("checkpoint experiment manifest reference does not match current run")
        if payload.get("evaluation_protocol_reference") != expected_evaluation_protocol_reference:
            raise ValueError("checkpoint evaluation protocol reference does not match current run")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload["scheduler"] is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload["amp_scaler"] is not None:
            scaler.load_state_dict(payload["amp_scaler"])
        _restore_rng(payload["rng"])
        return payload
