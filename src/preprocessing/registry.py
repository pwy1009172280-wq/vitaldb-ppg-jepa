"""Explicit stateless/fitted preprocessing contracts and registry."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

from src.data.access import SplitContext
from src.data.samples import UnifiedSample

PreprocessingTransform: TypeAlias = "StatelessTransform | FittedTransform"

_STAGES = ("record", "window")


class TransformNotFoundError(KeyError):
    """Raised when a configured preprocessing transform is not registered."""


class StatelessTransform(ABC):
    """Transform whose application requires no data-derived state."""

    version = "1"
    stage = "window"  # declared application stage: "record" or "window"

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def apply(self, sample: UnifiedSample) -> UnifiedSample:
        """Apply the transform to one sample."""

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "kind": "stateless", "stage": self.stage}


class FittedTransform(ABC):
    """Transform that can only fit on a training split context."""

    version = "1"
    stage = "window"  # declared application stage: "record" or "window"

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

    def state(self) -> dict[str, Any]:
        """Return the fitted state (serializable, for persistence/hashing)."""
        return {key: value for key, value in vars(self).items() if not key.startswith("_")}

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": "fitted",
            "stage": self.stage,
            "fitted": self._fitted,
            "state": self.state() if self._fitted else None,
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
    """Ordered transform pipeline with explicit train-only sequential fitting."""

    def __init__(self, transforms: tuple[StatelessTransform | FittedTransform, ...]) -> None:
        self.transforms = transforms

    def fit(self, samples: Iterable[UnifiedSample], *, context: SplitContext) -> "PreprocessingPipeline":
        """Fit transforms sequentially on the training split.

        The k-th fitted transform sees the output of all preceding (already
        fitted/stateless) transforms, never the raw input stream.
        """
        current = tuple(samples)
        for transform in self.transforms:
            if isinstance(transform, FittedTransform):
                transform.fit(current, context=context)
            current = tuple(self._apply_one(transform, sample) for sample in current)
        return self

    @staticmethod
    def _apply_one(transform: PreprocessingTransform, sample: UnifiedSample) -> UnifiedSample:
        result = transform.apply(sample)
        if not isinstance(result, UnifiedSample):
            raise TypeError("preprocessing transforms must return UnifiedSample")
        return result

    def apply(self, sample: UnifiedSample) -> UnifiedSample:
        current = sample
        for transform in self.transforms:
            current = self._apply_one(transform, current)
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
        if transform.stage not in _STAGES:
            raise ValueError(f"transform {name!r} has invalid stage {transform.stage!r}")
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
