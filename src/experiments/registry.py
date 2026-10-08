"""Registry/interface for experiment definitions, without execution logic."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class ExperimentInterface(ABC):
    """Minimal future-facing experiment contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable experiment name used for run directories."""

    @abstractmethod
    def build_config(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return resolved configuration; execution is intentionally out of scope."""


class ExperimentRegistry:
    """Explicit registry for experiment definitions or factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        if not name:
            raise ValueError("experiment name must be non-empty")
        if name in self._factories:
            raise ValueError(f"experiment already registered: {name}")
        self._factories[name] = factory

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._factories[name]
        except KeyError as error:
            raise KeyError(f"unknown experiment: {name}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

