"""Small, strict configuration schema shared by pipeline entry points."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    dataset: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    experiment: dict[str, Any] = field(default_factory=dict)

