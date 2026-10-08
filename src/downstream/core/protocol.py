"""Downstream-specific policy metadata around the authoritative protocol."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DownstreamPolicy:
    reader_name: str
    reader_version: str
    pooling_name: str
    pooling_version: str = "1"
    pretraining_overlap_policy: str = "validation_test_disjoint"
    checkpoint_selection: str = "last"
    selection_metric: str | None = None
    selection_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.reader_name or not self.reader_version or self.pooling_name not in ("mean", "reader_provided"):
            raise ValueError("reader and pooling policy are required")
        if self.checkpoint_selection not in ("last", "best_validation"):
            raise ValueError("checkpoint selection must be last or best_validation")
        if self.checkpoint_selection == "best_validation" and (not self.selection_metric or self.selection_mode not in ("min", "max")):
            raise ValueError("best_validation requires selection_metric and mode")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reader_name": self.reader_name,
            "reader_version": self.reader_version,
            "pooling_name": self.pooling_name,
            "pooling_version": self.pooling_version,
            "pretraining_overlap_policy": self.pretraining_overlap_policy,
            "checkpoint_selection": self.checkpoint_selection,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
        }
