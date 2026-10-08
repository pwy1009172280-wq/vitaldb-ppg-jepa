"""Structured training metric recording."""

import hashlib
import json
from pathlib import Path
from typing import Any


def protocol_hash(protocol_reference: str | None) -> str | None:
    if protocol_reference is None:
        return None
    return hashlib.sha256(protocol_reference.encode("utf-8")).hexdigest()


class MetricsLogger:
    def __init__(self, path: str | Path | None = None, protocol_reference: str | None = None) -> None:
        if not protocol_reference:
            raise ValueError("protocol_reference is required")
        self.path = Path(path) if path else None
        self.protocol_reference = protocol_reference
        self.records: list[dict[str, Any]] = self._load_existing()

    def _load_existing(self) -> list[dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return []
        import json

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw == {}:
            return []
        if not isinstance(raw, list):
            raise ValueError("metrics file must contain a list of records")
        expected_hash = protocol_hash(self.protocol_reference)
        for record in raw:
            if (
                record.get("protocol_reference") != self.protocol_reference
                or record.get("protocol_hash") != expected_hash
            ):
                raise ValueError("existing metrics protocol does not match current experiment")
        return raw

    def log(self, phase: str, epoch: int, global_step: int, terms: dict[str, float], batches: int) -> None:
        record = {
            "phase": phase,
            "epoch": epoch,
            "global_step": global_step,
            "loss_terms": dict(terms),
            "aggregation": {"level": "batch_mean", "num_batches": batches},
            "protocol_reference": self.protocol_reference,
            "protocol_hash": protocol_hash(self.protocol_reference),
        }
        self.records.append(record)
        self.flush()

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2) + "\n", encoding="utf-8")
