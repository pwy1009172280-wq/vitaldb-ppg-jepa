"""Model and structured-loss contracts for the generic training layer."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import torch


@dataclass(frozen=True)
class LossOutput:
    """A scalar optimization target plus named diagnostics."""

    total_loss: torch.Tensor
    loss_terms: Mapping[str, torch.Tensor | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.total_loss, torch.Tensor):
            raise TypeError("total_loss must be a torch.Tensor")
        if self.total_loss.ndim != 0:
            raise ValueError("total_loss must be scalar")
        if "total_loss" in self.loss_terms:
            raise ValueError("loss_terms cannot contain reserved name 'total_loss'")

    def detached_terms(self) -> dict[str, float]:
        terms: dict[str, float] = {"total_loss": float(self.total_loss.detach().cpu())}
        for name, value in self.loss_terms.items():
            if isinstance(value, torch.Tensor):
                if value.ndim != 0:
                    raise ValueError(f"loss term {name!r} must be scalar")
                terms[name] = float(value.detach().cpu())
            else:
                terms[name] = float(value)
        return terms


class TrainingModel(Protocol):
    """Model contract intentionally independent of datasets and methods."""

    def forward(self, batch: Any) -> Any:
        ...

    def compute_loss(self, output: Any, batch: Any) -> LossOutput:
        ...

    def on_optimizer_step(self) -> None:
        """Update optional target/EMA state after every optimizer step."""
