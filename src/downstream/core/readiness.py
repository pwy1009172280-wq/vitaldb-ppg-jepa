"""Evaluation readiness states without inventing a scientific protocol."""

from dataclasses import dataclass
from typing import Any

# Outcome states (fatal blockers -> never ready)
DATA_BLOCKED = "DATA_BLOCKED"
PROTOCOL_BLOCKED = "PROTOCOL_BLOCKED"
READY_FOR_EVALUATION = "READY_FOR_EVALUATION"

# Legacy alias kept for older consumers/audit tables; the assessor no longer
# returns it, but the string remains stable for backward reading.
DATA_READY_FOR_EVALUATION = "DATA_READY_FOR_EVALUATION"


@dataclass(frozen=True)
class EvaluationReadiness:
    state: str
    protocol_blockers: tuple[str, ...] = ()
    plumbing_blockers: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.state == READY_FOR_EVALUATION

    @property
    def protocol_blocked(self) -> bool:
        return bool(self.protocol_blockers)

    @property
    def data_blocked(self) -> bool:
        return bool(self.plumbing_blockers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "protocol_blockers": list(self.protocol_blockers),
            "plumbing_blockers": list(self.plumbing_blockers),
        }


def assess_evaluation_readiness(
    *,
    reader_ok: bool,
    labels_ok: bool,
    identities_ok: bool,
    adapter_ok: bool,
    protocol: Any | None = None,
    target_ref: str | None = None,
    split_ref: str | None = None,
    metric_config: tuple[str, ...] | None = None,
    aggregation: str | None = None,
) -> EvaluationReadiness:
    """Classify plumbing separately from protocol completeness (fail closed).

    Plumbing failures -> DATA_BLOCKED; data capable but no scientific protocol
    -> PROTOCOL_BLOCKED; only an explicit, complete protocol -> READY.
    The caller must provide protocol references; this function never selects a
    target, split, metric, or aggregation on its own.
    """

    plumbing = tuple(
        name for name, passed in (
            ("reader", reader_ok),
            ("labels", labels_ok),
            ("identities", identities_ok),
            ("task_adapter", adapter_ok),
        ) if not passed
    )
    if plumbing:
        return EvaluationReadiness(DATA_BLOCKED, plumbing_blockers=plumbing)
    protocol_blockers = []
    if protocol is None:
        protocol_blockers.append("protocol")
    if not target_ref:
        protocol_blockers.append("target")
    if not split_ref:
        protocol_blockers.append("split")
    if not metric_config:
        protocol_blockers.append("metric")
    if not aggregation:
        protocol_blockers.append("aggregation")
    if protocol_blockers:
        return EvaluationReadiness(PROTOCOL_BLOCKED, tuple(protocol_blockers))
    return EvaluationReadiness(READY_FOR_EVALUATION)
