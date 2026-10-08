"""Explicit stateless/fitted preprocessing contracts and registry."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

from src.data.access import SplitContext
from src.data.samples import UnifiedSample

PreprocessingTransform: TypeAlias = "StatelessTransform | FittedTransform"


class TransformNotFoundError(KeyError):
    """Raised when a configured preprocessing transform is not registered."""


class StatelessTransform(ABC):
    """Transform whose application requires no data-derived state."""

    version = "1"

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def apply(self, sample: UnifiedSample) -> UnifiedSample:
        """Apply the transform to one sample."""

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "kind": "stateless"}


class FittedTransform(ABC):
    """Transform that can only fit on a training split context."""

    version = "1"

    def __init__(self) -> None:
        self._fitted = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def fit(self, samples: Iterable[UnifiedSample], *, context: SplitContext) -> "FittedTransform":
        if context is None or context.role != "train":
            raise ValueError("fitted preprocessing may only be fit with train SplitContext")
        training_samples = tuple(samples)
        for sample in training_samples:
            if sample.subject_id not in context.subject_ids:
                raise ValueError(
                    f"sample subject {sample.subject_id!r} is outside the training split"
                )
        self._fit(training_samples)
        self._fitted = True
        return self

    @abstractmethod
    def _fit(self, samples: tuple[UnifiedSample, ...]) -> None:
        """Compute state from validated training samples only."""

    @abstractmethod
    def _apply(self, sample: UnifiedSample) -> UnifiedSample:
        """Apply the fitted state to one sample."""

    def apply(self, sample: UnifiedSample) -> UnifiedSample:
        if not self._fitted:
            raise RuntimeError(f"transform {self.name!r} must be fit before apply")
        return self._apply(sample)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": "fitted",
            "fitted": self._fitted,
        }


class _CallableStatelessTransform(StatelessTransform):
    """Compatibility adapter for the v1.1 callable registration API."""

    def __init__(self, transform: Callable[[UnifiedSample], UnifiedSample], name: str) -> None:
        self._transform = transform
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def apply(self, sample: UnifiedSample) -> UnifiedSample:
        return self._transform(sample)


class PreprocessingPipeline:
    """Ordered transform pipeline with explicit train-only fitting."""

    def __init__(self, transforms: tuple[StatelessTransform | FittedTransform, ...]) -> None:
        self.transforms = transforms

    def fit(self, samples: Iterable[UnifiedSample], *, context: SplitContext) -> "PreprocessingPipeline":
        training_samples = tuple(samples)
        for transform in self.transforms:
            if isinstance(transform, FittedTransform):
                transform.fit(training_samples, context=context)
        return self

    def apply(self, sample: UnifiedSample) -> UnifiedSample:
        current = sample
        for transform in self.transforms:
            current = transform.apply(current)
            if not isinstance(current, UnifiedSample):
                raise TypeError("preprocessing transforms must return UnifiedSample")
        return current

    def __call__(self, sample: UnifiedSample) -> UnifiedSample:
        return self.apply(sample)

    def metadata(self) -> list[dict[str, Any]]:
        return [transform.metadata() for transform in self.transforms]


class PreprocessingRegistry:
    """Name-to-contract registry; no concrete signal transforms are included."""

    def __init__(self) -> None:
        self._transforms: dict[str, StatelessTransform | FittedTransform] = {}

    def register(self, name: str, transform: StatelessTransform | FittedTransform | Callable) -> None:
        if not name:
            raise ValueError("transform name must be non-empty")
        if name in self._transforms:
            raise ValueError(f"preprocessing transform already registered: {name}")
        if callable(transform) and not isinstance(transform, (StatelessTransform, FittedTransform)):
            transform = _CallableStatelessTransform(transform, name)
        if not isinstance(transform, (StatelessTransform, FittedTransform)):
            raise TypeError("transform must implement StatelessTransform or FittedTransform")
        self._transforms[name] = transform

    def get(self, name: str) -> StatelessTransform | FittedTransform:
        try:
            return self._transforms[name]
        except KeyError as error:
            raise TransformNotFoundError(name) from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._transforms))

    def compose(self, names: Iterable[str]) -> PreprocessingPipeline:
        return PreprocessingPipeline(tuple(self.get(name) for name in names))
