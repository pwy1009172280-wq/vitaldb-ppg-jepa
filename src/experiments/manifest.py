"""Run-level experiment manifest."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .metadata import serialize_config
from .protocol import EvaluationProtocol


@dataclass(frozen=True)
class ExperimentRunManifest:
    """Complete provenance record for one experiment run."""

    experiment_name: str
    resolved_config: dict[str, Any]
    dataset_manifest_ref: str
    subject_split_ref: str
    seed: int
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    code_version: str | None = None
    preprocessing_version: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    git_state: str | None = None
    evaluation_protocol: EvaluationProtocol | None = None
    pretraining_subject_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.experiment_name or not self.dataset_manifest_ref or not self.subject_split_ref:
            raise ValueError("experiment name and manifest references are required")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "resolved_config", serialize_config(self.resolved_config))
        if len(set(self.pretraining_subject_ids)) != len(self.pretraining_subject_ids):
            raise ValueError("pretraining_subject_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "resolved_config": self.resolved_config,
            "dataset_manifest_ref": self.dataset_manifest_ref,
            "subject_split_ref": self.subject_split_ref,
            "seed": self.seed,
            "timestamp_utc": self.timestamp_utc,
            "code_version": self.code_version,
            "preprocessing_version": self.preprocessing_version,
            "environment": dict(self.environment),
            "git_commit": self.git_commit,
            "git_state": self.git_state,
            "evaluation_protocol": (
                self.evaluation_protocol.to_dict() if self.evaluation_protocol else None
            ),
            "pretraining_subject_ids": list(self.pretraining_subject_ids),
        }
