"""Evaluation protocol metadata, without evaluation algorithms."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationProtocol:
    task: str
    protocol_type: str
    metrics: tuple[str, ...]
    aggregation_level: str
    backbone_frozen: bool

    def __post_init__(self) -> None:
        if not self.task or not self.protocol_type or not self.aggregation_level:
            raise ValueError("task, protocol_type, and aggregation_level are required")
        if not self.metrics:
            raise ValueError("at least one metric is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "protocol_type": self.protocol_type,
            "metrics": list(self.metrics),
            "aggregation_level": self.aggregation_level,
            "backbone_frozen": self.backbone_frozen,
        }

