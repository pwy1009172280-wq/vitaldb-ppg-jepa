"""Local layer-probe schema; the existing PipelineConfig is unchanged."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
import re

import yaml


SUPPORTED_METRICS = {
    "classification": {"accuracy", "macro_f1"},
    "regression": {"mae", "rmse"},
}


def validate_layers(layers: tuple[str, ...]) -> None:
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must be non-empty and unique")
    if any(not isinstance(name, str) or not re.fullmatch(r"layer_[1-9][0-9]*|final_layer", name) for name in layers):
        raise ValueError("layers must be layer_1, layer_2, ... or final_layer")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    type: str = "layer_probe"
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.name in (".", "..") or "/" in self.name or "\\" in self.name:
            raise ValueError("experiment name must be a safe directory name")
        if self.type != "layer_probe":
            raise ValueError("only layer_probe is supported")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class EncoderConfig:
    checkpoint: str
    layers: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.checkpoint, (str, Path)) or not str(self.checkpoint):
            raise ValueError("encoder checkpoint is required")
        if not isinstance(self.layers, (list, tuple)):
            raise ValueError("layers must be a list or tuple of layer names")
        object.__setattr__(self, "checkpoint", str(self.checkpoint))
        object.__setattr__(self, "layers", tuple(self.layers))
        validate_layers(self.layers)


@dataclass(frozen=True)
class DownstreamConfig:
    task: str
    target: str
    metric: str

    def __post_init__(self):
        if not isinstance(self.task, str) or self.task not in SUPPORTED_METRICS:
            raise ValueError("task must be classification or regression")
        if not isinstance(self.target, str) or not self.target or not isinstance(self.metric, str) or self.metric not in SUPPORTED_METRICS[self.task]:
            raise ValueError("target and a supported task metric are required")


@dataclass(frozen=True)
class ProbeConfig:
    epochs: int = 10
    batch_size: int = 32
    standardize: bool = True
    optimizer: dict[str, Any] = field(default_factory=lambda: {"name": "sgd", "kwargs": {"lr": 0.01}})
    checkpoint_selection: str = "last"

    def __post_init__(self):
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (self.epochs, self.batch_size)):
            raise ValueError("epochs and batch_size must be positive integers")
        if not isinstance(self.standardize, bool):
            raise ValueError("standardize must be boolean")
        if not isinstance(self.optimizer, dict) or set(self.optimizer) != {"name", "kwargs"}:
            raise ValueError("optimizer requires name and kwargs")
        if not isinstance(self.optimizer["name"], str) or not self.optimizer["name"] or not isinstance(self.optimizer["kwargs"], dict):
            raise ValueError("invalid optimizer configuration")
        if self.checkpoint_selection != "last":
            raise ValueError("phase 1 supports explicit last selection only")


@dataclass(frozen=True)
class LayerProbeConfig:
    experiment: ExperimentConfig
    encoder: EncoderConfig
    downstream: DownstreamConfig
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LayerProbeConfig":
        if not isinstance(raw, Mapping) or set(raw) - {"experiment", "encoder", "downstream", "probe"}:
            raise ValueError("unknown layer-probe config sections or invalid root")
        try:
            return cls(
                ExperimentConfig(**raw["experiment"]), EncoderConfig(**raw["encoder"]),
                DownstreamConfig(**raw["downstream"]), ProbeConfig(**raw.get("probe", {})),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid layer-probe config: {error}") from error


def load_layer_probe_config(path: str | Path) -> LayerProbeConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return LayerProbeConfig.from_dict(raw)
