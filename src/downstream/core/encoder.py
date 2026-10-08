"""Small model-independent representation and freezing helpers."""

from dataclasses import dataclass
from typing import Literal

import torch


RepresentationKind = Literal["token", "vector"]


@dataclass(frozen=True)
class Representation:
    """A named reader output with an explicit tensor layout."""

    tensor: torch.Tensor
    kind: RepresentationKind

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("representation tensor must be a torch.Tensor")
        if self.kind not in ("token", "vector"):
            raise ValueError("representation kind must be token or vector")
        expected_ndim = 3 if self.kind == "token" else 2
        if self.tensor.ndim != expected_ndim:
            raise ValueError(f"{self.kind} representations must have {expected_ndim} dimensions")
        if self.tensor.shape[0] == 0 or self.tensor.shape[-1] == 0:
            raise ValueError("representation must have non-empty batch and feature dimensions")


def freeze_encoder(model: torch.nn.Module) -> torch.nn.Module:
    """Put an extraction model in eval/no-grad-parameter state."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
