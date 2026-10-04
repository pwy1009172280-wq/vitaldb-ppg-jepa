"""Serializable contracts for dataset and experiment manifests."""

from dataclasses import dataclass, field
from typing import Any

from .leakage import assert_manifest_splits_disjoint
from .splits import SubjectSplit


@dataclass(frozen=True)
class DatasetManifest:
    """Metadata-only description of an indexed dataset; it never loads data."""

    name: str
    version: str
    modality: str
    subject_ids: tuple[str, ...]
    recording_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    preprocessing_version: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    git_state: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.version or not self.modality:
            raise ValueError("name, version, and modality must be non-empty")
        if not self.subject_ids or any(not subject for subject in self.subject_ids):
            raise ValueError("subject_ids must contain non-empty ids")
        if len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("subject_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "modality": self.modality,
            "subject_ids": list(self.subject_ids),
            "recording_ids": list(self.recording_ids),
            "metadata": dict(self.metadata),
            "preprocessing_version": self.preprocessing_version,
            "environment": dict(self.environment),
            "git_commit": self.git_commit,
            "git_state": self.git_state,
        }


@dataclass(frozen=True)
class ExperimentManifest:
    """Provenance contract tying datasets, subject split, and task together."""

    name: str
    task: str
    dataset_names: tuple[str, ...]
    split: SubjectSplit
    seed: int
    config: dict[str, Any] = field(default_factory=dict)
    preprocessing_version: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    git_state: str | None = None
    evaluation_protocol_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.task or not self.dataset_names:
            raise ValueError("name, task, and dataset_names are required")
        if len(set(self.dataset_names)) != len(self.dataset_names):
            raise ValueError("dataset_names must be unique")
        assert_manifest_splits_disjoint(self.split.as_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "dataset_names": list(self.dataset_names),
            "split": self.split.as_dict(),
            "seed": self.seed,
            "config": dict(self.config),
            "preprocessing_version": self.preprocessing_version,
            "environment": dict(self.environment),
            "git_commit": self.git_commit,
            "git_state": self.git_state,
            "evaluation_protocol_ref": self.evaluation_protocol_ref,
        }
