"""Thin task boundary between source samples and downstream evaluation.

Dataset readers expose source labels in :class:`UnifiedSample`.  This module
does not define a scientific task; it only requires an explicitly supplied
target name and carries its provenance into the evaluator.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ...data.samples import UnifiedSample


@dataclass(frozen=True)
class TaskTarget:
    """A target value with the identity and provenance needed by evaluation."""

    value: Any
    sample_id: str
    subject_id: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.subject_id:
            raise ValueError("task targets require sample and subject identity")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("task target provenance must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "provenance": dict(self.provenance),
        }


class TaskAdapter(Protocol):
    """Minimal adapter contract; target semantics live in its configuration."""

    name: str
    version: str
    target_name: str

    def adapt(self, sample: UnifiedSample | Mapping[str, Any]) -> TaskTarget:
        ...

    def to_dict(self) -> dict[str, Any]:
        ...


def _field(sample: UnifiedSample | Mapping[str, Any], name: str) -> Any:
    if isinstance(sample, Mapping):
        try:
            return sample[name]
        except KeyError as error:
            raise ValueError(f"feature sample is missing {name!r}") from error
    return getattr(sample, name)


def _sample_id(sample: UnifiedSample | Mapping[str, Any]) -> str:
    dataset = str(_field(sample, "dataset"))
    recording = str(_field(sample, "recording_id"))
    window = _field(sample, "window_id")
    if window is not None:
        return f"{dataset}/{recording}/{window}"
    start = _field(sample, "window_start_sample")
    end = _field(sample, "window_end_sample")
    if start is not None or end is not None:
        return f"{dataset}/{recording}/{start}:{end}"
    return f"{dataset}/{recording}/{_field(sample, 'start_time_s')}:{_field(sample, 'end_time_s')}"


@dataclass(frozen=True)
class LabelTaskAdapter:
    """Map one explicitly named source label to a task target.

    ``target_name`` is mandatory by design.  There is no dataset-specific
    default target and no implicit label-to-task conversion.
    """

    target_name: str
    name: str = "source_label"
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.target_name:
            raise ValueError("target_name must be explicit")

    def adapt(self, sample: UnifiedSample | Mapping[str, Any]) -> TaskTarget:
        labels = _field(sample, "labels")
        if not isinstance(labels, Mapping) or self.target_name not in labels:
            raise ValueError(f"sample is missing target label {self.target_name!r}")
        value = labels[self.target_name]
        if value is None:
            raise ValueError(f"target label {self.target_name!r} is null")
        source_provenance = _field(sample, "provenance")
        provenance = {
            "adapter": self.name,
            "adapter_version": self.version,
            "target_name": self.target_name,
            "label_source": "sample.labels",
            "source_provenance": dict(source_provenance),
        }
        return TaskTarget(
            value=value,
            sample_id=_sample_id(sample),
            subject_id=str(_field(sample, "subject_id")),
            provenance=provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "target_name": self.target_name}


def adapt_targets(
    samples: Sequence[UnifiedSample | Mapping[str, Any]], adapter: TaskAdapter
) -> tuple[TaskTarget, ...]:
    """Adapt samples in order while retaining one target per source sample."""

    return tuple(adapter.adapt(sample) for sample in samples)
