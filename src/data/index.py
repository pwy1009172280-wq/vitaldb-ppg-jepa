"""Small, dependency-light canonical record-index utilities.

The index is metadata only.  It deliberately uses JSON Lines because this
repository has no existing Parquet dependency or index standard; callers can
convert it later without changing the reader contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class RecordIndexRow:
    dataset: str
    dataset_version: str
    subject_id: str
    record_id: str
    segment_id: str | None
    session_id: str | None
    modality: str
    channel_name: tuple[str, ...]
    sampling_rate_hz: tuple[float | None, ...]
    unit: tuple[str, ...]
    n_samples: int | None
    duration_s: float | None
    source_path: str
    source_format: str
    source_variant: str
    continuity: str
    provenance: str
    role_reference: str
    qc_status: str
    qc_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(rows: Iterable[RecordIndexRow], path: str | Path) -> int:
    """Write deterministic metadata rows and return the number of rows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[RecordIndexRow]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                payload["channel_name"] = tuple(payload["channel_name"])
                payload["sampling_rate_hz"] = tuple(payload["sampling_rate_hz"])
                payload["unit"] = tuple(payload["unit"])
                yield RecordIndexRow(
                    **payload,
                )
            except Exception as exc:
                raise ValueError(f"invalid index row {line_number}: {exc}") from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_hash(path: str | Path) -> Path:
    target = Path(path)
    hash_path = target.with_suffix(target.suffix + ".sha256")
    hash_path.write_text(f"{sha256_file(target)}  {target.name}\n", encoding="utf-8")
    return hash_path
