"""PTB-XL source-native index builder and reader.

No resampling or windowing is performed.  PTB-XL's official 100 Hz and
500 Hz representations remain separate index rows and reader variants.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from src.data.base import BaseDataset
from src.data.index import RecordIndexRow, read_jsonl, write_hash, write_jsonl
from src.data.samples import UnifiedSample


DATASET = "ptb-xl"
VERSION = "1.0.3"
PROVENANCE = "PhysioNet official PTB-XL v1.0.3"
ROLE_REFERENCE = "registry/role_policy.yaml"


def _stable_patient_id(value: str) -> str:
    value = value.strip()
    if value.endswith(".0"):
        value = value[:-2]
    return f"{DATASET}:{value}"


def _header_metadata(path: Path) -> tuple[int, float, int, tuple[str, ...], tuple[str, ...]]:
    """Parse the stable WFDB header fields without loading waveform data."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip().split()
        if len(first) < 3:
            raise ValueError(f"invalid WFDB header: {path}")
        n_channels = int(first[1])
        fs = float(first[2].split("/")[0])
        n_samples = int(first[3]) if len(first) > 3 else 0
        names: list[str] = []
        units: list[str] = []
        for _ in range(n_channels):
            fields = handle.readline().strip().split()
            if not fields:
                break
            names.append(fields[-1] if len(fields) >= 1 else "UNKNOWN")
            units.append("UNKNOWN")
    if len(names) != n_channels:
        names.extend([f"channel_{i}" for i in range(len(names), n_channels)])
        units.extend(["UNKNOWN"] * (n_channels - len(units)))
    return n_channels, fs, n_samples, tuple(names), tuple(units)


def _read_wfdb_without_wfdb(header: Path, n_channels: int, n_samples: int) -> tuple[np.ndarray, float, tuple[str, ...], tuple[str, ...]]:
    """Read the PTB-XL 16-bit source format without adding a new dependency."""
    import numpy as np

    lines = header.read_text(encoding="utf-8", errors="replace").splitlines()
    first = lines[0].split()
    fs = float(first[2].split("/")[0])
    names: list[str] = []
    units: list[str] = []
    gains: list[float] = []
    baselines: list[float] = []
    for line in lines[1 : n_channels + 1]:
        fields = line.split()
        if len(fields) < 3 or int(fields[1]) != 16:
            raise RuntimeError(f"unsupported source-native WFDB format in {header}")
        match = re.match(r"([^(/]+)\(([^)]*)\)(?:/([^ ]+))?", fields[2])
        if not match:
            raise RuntimeError(f"cannot parse gain/baseline in {header}: {fields[2]}")
        gains.append(float(match.group(1)))
        baselines.append(float(match.group(2) or 0))
        units.append(match.group(3) or "UNKNOWN")
        names.append(fields[-1] if fields[-1] else "UNKNOWN")
    dat = np.fromfile(header.with_suffix(".dat"), dtype="<i2")
    expected = n_channels * n_samples
    if dat.size < expected:
        raise RuntimeError(f"truncated WFDB data file: {header.with_suffix('.dat')}")
    digital = dat[:expected].reshape(n_samples, n_channels).astype(np.float32)
    physical = (digital - np.asarray(baselines, dtype=np.float32)) / np.asarray(gains, dtype=np.float32)
    return physical.T, fs, tuple(names), tuple(units)


def build_index(root: str | Path, output: str | Path | None = None) -> Path:
    root = Path(root)
    source_root = root / "incoming" / "ptb-xl-1.0.3"
    metadata_path = source_root / "ptbxl_database.csv"
    if output is None:
        output = root / "metadata" / "records.jsonl"
    output = Path(output)
    rows: list[RecordIndexRow] = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            subject = _stable_patient_id(record["patient_id"])
            record_id = record["ecg_id"].strip()
            for variant, field in (("lr_100hz", "filename_lr"), ("hr_500hz", "filename_hr")):
                rel_base = record[field].strip()
                header_rel = rel_base + ".hea"
                header_path = source_root / header_rel
                n_channels, fs, n_samples, names, units = _header_metadata(header_path)
                rows.append(
                    RecordIndexRow(
                        dataset=DATASET,
                        dataset_version=VERSION,
                        subject_id=subject,
                        record_id=f"{record_id}:{variant}",
                        segment_id=None,
                        session_id=None,
                        modality="ECG",
                        channel_name=names,
                        sampling_rate_hz=(fs,) * n_channels,
                        unit=units,
                        n_samples=n_samples,
                        duration_s=(n_samples / fs if n_samples else None),
                        source_path=f"incoming/ptb-xl-1.0.3/{header_rel}",
                        source_format="WFDB",
                        source_variant=variant,
                        continuity="source-native; no gap repair",
                        provenance=PROVENANCE,
                        role_reference=ROLE_REFERENCE,
                        qc_status="PASS",
                        qc_reason=None,
                    )
                )
    write_jsonl(rows, output)
    write_hash(output)
    return output


class PTBXLReader(BaseDataset):
    """Read one source-native PTB-XL representation as a UnifiedSample."""

    name = DATASET
    version = VERSION

    def __init__(self, root: str | Path, index_path: str | Path | None = None):
        self.root = Path(root)
        self.source_root = self.root / "incoming" / "ptb-xl-1.0.3"
        self.index_path = Path(index_path or self.root / "metadata" / "records.jsonl")
        self.rows = tuple(read_jsonl(self.index_path))

    def __len__(self) -> int:
        return len(self.rows)

    def subject_ids(self) -> frozenset[str]:
        return frozenset(row.subject_id for row in self.rows)

    def __getitem__(self, index: int) -> UnifiedSample:
        row = self.rows[index]
        return self.read_record(row)

    def read_record(self, row: RecordIndexRow) -> UnifiedSample:
        import numpy as np

        header = self.root / row.source_path
        try:
            import wfdb
        except ImportError:
            signal, fs, names, units = _read_wfdb_without_wfdb(header, len(row.channel_name), row.n_samples or 0)
        else:
            record = wfdb.rdrecord(str(header.with_suffix("")))
            signal = np.asarray(record.p_signal, dtype=np.float32).T
            fs = float(record.fs)
            names = tuple(str(x) for x in (record.sig_name or row.channel_name))
            units = tuple(str(x) if x else "UNKNOWN" for x in (getattr(record, "units", None) or row.unit))
        return UnifiedSample(
            signal=signal,
            subject_id=row.subject_id,
            recording_id=row.record_id,
            dataset=DATASET,
            modality="ECG",
            sampling_rate_hz=fs,
            start_time_s=0.0,
            end_time_s=signal.shape[1] / fs,
            channel_names=names,
            units=units,
            provenance={
                "source_path": row.source_path,
                "source_format": row.source_format,
                "source_variant": row.source_variant,
                "dataset_version": VERSION,
                "reader": "PTBXLReader",
                "reader_version": "1.0",
            },
            metadata={"continuity": row.continuity, "role_reference": row.role_reference},
        )
