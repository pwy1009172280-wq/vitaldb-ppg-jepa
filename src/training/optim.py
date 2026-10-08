"""Config-driven optimizer and scheduler factories."""

from collections.abc import Callable, Iterable
from typing import Any

import torch


class OptimizerFactory:
    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., torch.optim.Optimizer]] = {}

    def register(self, name: str, builder: Callable[..., torch.optim.Optimizer]) -> None:
        if not name or name in self._builders:
            raise ValueError(f"invalid or duplicate optimizer: {name!r}")
        self._builders[name] = builder

    def create(self, name: str, parameters: Iterable[torch.nn.Parameter], **config: Any) -> torch.optim.Optimizer:
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

