"""Config-driven optimizer and scheduler factories.

``OptimizerFactory.create`` accepts either a flat iterable of parameters or an
explicit list of parameter-group dicts (e.g. from ``adamw_parameter_groups``);
it passes them through to the registered builder unchanged. Formal parameter
grouping *policy* remains a Research Gate decision; the helper below is the
legacy grouping capability reused as a parity reference.
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch

from src.pretrain.optim import adamw_parameter_groups

__all__ = ["OptimizerFactory", "SchedulerFactory", "adamw_parameter_groups"]


class OptimizerFactory:
    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., torch.optim.Optimizer]] = {}

    def register(self, name: str, builder: Callable[..., torch.optim.Optimizer]) -> None:
        if not name or name in self._builders:
            raise ValueError(f"invalid or duplicate optimizer: {name!r}")
        self._builders[name] = builder

    def create(self, name: str, parameters: Iterable[torch.nn.Parameter] | list[dict[str, Any]], **config: Any) -> torch.optim.Optimizer:
        """Build an optimizer for a flat parameter iterable or explicit groups."""
        try:
            return self._builders[name](parameters, **config)
        except KeyError as error:
            raise KeyError(f"unknown optimizer: {name}") from error


class SchedulerFactory:
    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, builder: Callable[..., Any]) -> None:
        if not name or name in self._builders:
            raise ValueError(f"invalid or duplicate scheduler: {name!r}")
        self._builders[name] = builder

    def create(self, name: str, optimizer: torch.optim.Optimizer, **config: Any) -> Any:
        try:
            return self._builders[name](optimizer, **config)
        except KeyError as error:
            raise KeyError(f"unknown scheduler: {name}") from error
