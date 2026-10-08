"""Source-native PPG-DaLiA index and reader.

The acquisition is retained as the official ZIP archive.  The reader extracts
only the requested subject pickle to a caller-provided temporary directory;
it never rewrites, aligns, resamples, or concatenates source signals.
"""

from __future__ import annotations

import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from src.data.base import BaseDataset
from src.data.index import RecordIndexRow, read_jsonl, write_hash, write_jsonl
from src.data.samples import (
    SUBJECT_IDENTITY_RESOLVED,
    SUBJECT_IDENTITY_UNRESOLVED,
    UnifiedSample,
)


DATASET = "ppg-dalia"
VERSION = "UCI release accessed 2026-10-03"
PROVENANCE = "UCI Machine Learning Repository dataset 495 / official PPG-DaLiA source"
ROLE_REFERENCE = "registry/role_policy.yaml"

# These are source-documented native representations, not experiment choices.
SIGNALS: tuple[tuple[str, str, str, tuple[str, ...], float], ...] = (
    ("wrist", "BVP", "PPG", ("BVP",), 64.0),
    ("wrist", "ACC", "ACC", ("ACC_X", "ACC_Y", "ACC_Z"), 32.0),
    ("wrist", "EDA", "EDA", ("EDA",), 4.0),
    ("wrist", "TEMP", "TEMP", ("TEMP",), 4.0),
    ("chest", "ECG", "ECG", ("ECG",), 700.0),
    ("chest", "EMG", "EMG", ("EMG",), 700.0),
    ("chest", "EDA", "EDA", ("EDA",), 700.0),
    ("chest", "TEMP", "TEMP", ("TEMP",), 700.0),
    ("chest", "ACC", "ACC", ("ACC_X", "ACC_Y", "ACC_Z"), 700.0),
    ("chest", "Resp", "RESP", ("RESP",), 700.0),
)


def _subject_members(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    result = []
    for name in archive.namelist():
        parts = Path(name).parts
        if len(parts) == 3 and parts[0] == "PPG_FieldStudy" and parts[1].startswith("S") and parts[2] == f"{parts[1]}.pkl":
            result.append((parts[1], name))
    return sorted(result, key=lambda x: int(x[0][1:]))


def build_index(root: str | Path, output: str | Path | None = None) -> Path:
    root = Path(root)
    archive_path = root / "data" / "data.zip"
    if output is None:
        output = root / "metadata" / "records.jsonl"
    output = Path(output)
    rows: list[RecordIndexRow] = []
    with zipfile.ZipFile(archive_path) as archive:
        subjects = _subject_members(archive)
    for subject, member in subjects:
        for device, signal_key, modality, channel_names, fs in SIGNALS:
            rows.append(
                RecordIndexRow(
                    dataset=DATASET,
                    dataset_version=VERSION,
                    subject_id=f"{DATASET}:{subject}",
                    record_id=f"{subject}:{device}_{signal_key}",
                    segment_id=None,
                    session_id=subject,
                    modality=modality,
                    channel_name=channel_names,
                    sampling_rate_hz=(fs,) * len(channel_names),
                    unit=("UNKNOWN",) * len(channel_names),
                    n_samples=None,
                    duration_s=None,
                    source_path=f"data/data.zip!{member}",
                    source_format="pickle-in-zip",
                    source_variant=f"{device}:{signal_key}",
                    continuity="source-native; timing and gaps not repaired",
                    provenance=PROVENANCE,
                    role_reference=ROLE_REFERENCE,
                    qc_status="UNKNOWN",
                    qc_reason="requires source-native pickle decode; no eager full-archive read",
                    subject_identity_status=SUBJECT_IDENTITY_RESOLVED,
                    subject_identity_namespace=DATASET,
                    subject_identity_kind="official_subject_id",
                    subject_source_identity=subject,
                )
            )
    write_jsonl(rows, output)
    write_hash(output)
    return output


def _extract_signal(payload: Any, device: str, key: str) -> np.ndarray:
    import numpy as np

    def lookup(mapping: dict[Any, Any], name: str) -> Any:
        return mapping.get(name, mapping.get(name.encode("ascii")))

    signal = lookup(payload, "signal") if isinstance(payload, dict) else payload
    device_data = lookup(signal, device) if isinstance(signal, dict) else None
    value_data = lookup(device_data, key) if isinstance(device_data, dict) else None
    if value_data is None:
        raise KeyError(f"source signal {device}:{key} not found")
    value = np.asarray(value_data)
    if value.ndim == 1:
        return value[np.newaxis, :]
    if value.ndim != 2:
        raise ValueError(f"unsupported signal shape {value.shape} for {device}:{key}")
    # Source ACC arrays are time x channels; preserve values, only expose the
    # existing source orientation as the canonical channels x time contract.
    return value.T if value.shape[0] >= value.shape[1] else value


class PPGDaLiAReader(BaseDataset):
    name = DATASET
    version = VERSION

    def __init__(self, root: str | Path, index_path: str | Path | None = None, scratch_dir: str | Path | None = None):
        self.root = Path(root)
        self.archive_path = self.root / "data" / "data.zip"
        self.index_path = Path(index_path or self.root / "metadata" / "records.jsonl")
        self.rows = tuple(read_jsonl(self.index_path))
        self.scratch_dir = Path(scratch_dir) if scratch_dir else None

    def __len__(self) -> int:
        return len(self.rows)

    def subject_ids(self) -> frozenset[str]:
        return frozenset(
            row.subject_id for row in self.rows
            if row.subject_identity_status == SUBJECT_IDENTITY_RESOLVED and row.subject_id
        )

    def __getitem__(self, index: int) -> UnifiedSample:
        return self.read_record(self.rows[index])

    def read_record(self, row: RecordIndexRow) -> UnifiedSample:
        import numpy as np

        member = row.source_path.split("!", 1)[1]
        device, key = row.source_variant.split(":", 1)
        scratch = self.scratch_dir or Path(tempfile.gettempdir())
        with tempfile.TemporaryDirectory(prefix="ppg_dalia_reader_", dir=scratch) as tmp:
            extracted = Path(tmp) / Path(member).name
            with zipfile.ZipFile(self.archive_path) as archive, archive.open(member) as source, extracted.open("wb") as target:
                target.write(source.read())
            with extracted.open("rb") as handle:
                payload = pickle.load(handle, encoding="latin1")
        signal = _extract_signal(payload, device, key).astype(np.float32, copy=False)
        fs = float(row.sampling_rate_hz[0] or 0)
        duration = signal.shape[1] / fs
        if row.subject_identity_status == SUBJECT_IDENTITY_RESOLVED and row.subject_id:
            subject_id = row.subject_id
            status = SUBJECT_IDENTITY_RESOLVED
            src_id = row.subject_source_identity
        else:
            subject_id = None
            status = SUBJECT_IDENTITY_UNRESOLVED
            src_id = row.subject_id or row.subject_source_identity
        return UnifiedSample(
            signal=signal,
            subject_id=subject_id,
            recording_id=row.record_id,
            dataset=DATASET,
            modality=row.modality,
            sampling_rate_hz=fs,
            start_time_s=0.0,
            end_time_s=duration,
            channel_names=row.channel_name,
            units=row.unit,
            provenance={
                "source_path": row.source_path,
                "source_format": row.source_format,
                "source_variant": row.source_variant,
                "dataset_version": VERSION,
                "reader": "PPGDaLiAReader",
                "reader_version": "1.0",
            },
            metadata={"session_id": row.session_id, "continuity": row.continuity, "role_reference": row.role_reference},
            subject_identity_status=status,
            subject_identity_namespace=row.subject_identity_namespace,
            subject_identity_kind=row.subject_identity_kind,
            subject_source_identity=src_id,
        )
