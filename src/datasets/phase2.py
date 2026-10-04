"""Source-native adapters for the approved Data Bank Phase 2 datasets.

The adapters intentionally expose records as-is.  They do not resample,
window, normalize, filter, or choose a model channel.
"""

from __future__ import annotations

import io
import pickle
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

from src.data.base import BaseDataset
from src.data.index import RecordIndexRow, read_jsonl, write_hash, write_jsonl
from src.data.samples import UnifiedSample

ROLE_REFERENCE = "registry/role_policy.yaml"


def _header(path: Path) -> tuple[int, float, int, tuple[str, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...], int]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    first = lines[0].split()
    nchan, fs = int(first[1]), float(first[2].split("/")[0])
    nsamp = int(first[3]) if len(first) > 3 else 0
    names: list[str] = []
    units: list[str] = []
    gains: list[float] = []
    baselines: list[float] = []
    fmts: list[int] = []
    for line in lines[1 : nchan + 1]:
        fields = line.split()
        fmt = fields[1].split("/")[0]
        fmts.append(int(fmt.split("x", 1)[0].split("+", 1)[0]))
        match = re.match(r"([^()]+)\(([^)]*)\)(?:/([^ ]+))?", fields[2])
        gains.append(float(match.group(1)) if match else 1.0)
        baselines.append(float(match.group(2) or 0.0) if match else 0.0)
        units.append(match.group(3) if match and match.group(3) else "UNKNOWN")
        names.append(fields[-1] if len(fields) > 8 else f"channel_{len(names)}")
    while len(names) < nchan:
        names.append(f"channel_{len(names)}")
        units.append("UNKNOWN")
        gains.append(1.0)
        baselines.append(0.0)
        fmts.append(16)
    return nchan, fs, nsamp, tuple(names), tuple(units), tuple(gains), tuple(baselines), fmts[0]


def _physical(digital: np.ndarray, gains: tuple[float, ...], baselines: tuple[float, ...]) -> np.ndarray:
    return (digital.astype(np.float32) - np.asarray(baselines, dtype=np.float32)[:, None]) / np.asarray(gains, dtype=np.float32)[:, None]


def _read_wfdb(header: Path, meta: tuple) -> tuple[np.ndarray, float, tuple[str, ...], tuple[str, ...]]:
    nchan, fs, nsamp, names, units, gains, baselines, fmt = meta
    dat = header.with_suffix(".dat").read_bytes()
    if fmt == 16:
        raw = np.frombuffer(dat, dtype="<i2", count=nchan * nsamp)
        digital = raw.reshape(nsamp, nchan).T
    elif fmt == 212 and nchan == 2:
        raw = np.frombuffer(dat, dtype=np.uint8)
        pairs = min((nsamp + 1) // 2, raw.size // 3)
        raw = raw[: pairs * 3].reshape(-1, 3)
        a = raw[:, 0].astype(np.int16) | ((raw[:, 1] & 0x0F).astype(np.int16) << 8)
        b = raw[:, 2].astype(np.int16) | ((raw[:, 1] >> 4).astype(np.int16) << 8)
        a[a >= 2048] -= 4096
        b[b >= 2048] -= 4096
        digital = np.vstack([a[:nsamp], b[:nsamp]])
    else:
        raise RuntimeError(f"unsupported WFDB format {fmt} in {header}")
    if digital.shape[1] < nsamp:
        raise RuntimeError(f"truncated WFDB data file: {header.with_suffix('.dat')}")
    return _physical(digital[:, :nsamp], gains, baselines), fs, names, units


def _read_mat(path: Path, header: Path) -> tuple[np.ndarray, float, tuple[str, ...], tuple[str, ...]]:
    meta = _header(header)
    values = loadmat(path)
    arrays = [(k, v) for k, v in values.items() if not k.startswith("__") and isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number)]
    if not arrays:
        raise RuntimeError(f"no numeric signal array in {path}")
    signal = next((v for k, v in arrays if k == "val"), arrays[0][1])
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise RuntimeError(f"expected 2-D MATLAB signal in {path}")
    if signal.shape[0] != meta[0] and signal.shape[1] == meta[0]:
        signal = signal.T
    if signal.shape[0] != meta[0]:
        raise RuntimeError(f"MATLAB channel count mismatch in {path}")
    return _physical(signal[:, : meta[2]], meta[5], meta[6]), meta[1], meta[3], meta[4]


class _Indexed(BaseDataset):
    name = ""
    version = ""

    def __init__(self, root: str | Path, index_path: str | Path | None = None):
        self.root = Path(root)
        self.index_path = Path(index_path or self.root / "metadata" / "records.jsonl")
        self.rows = tuple(read_jsonl(self.index_path))

    def __len__(self) -> int:
        return len(self.rows)

    def subject_ids(self) -> frozenset[str]:
        return frozenset(row.subject_id for row in self.rows)


class ECGReader(_Indexed):
    def __getitem__(self, index: int) -> UnifiedSample:
        row = self.rows[index]
        header = self.root / row.source_path
        if header.suffix == ".hea":
            signal, fs, names, units = _read_wfdb(header, _header(header))
        else:
            signal, fs, names, units = _read_mat(header, header.with_suffix(".hea"))
        return UnifiedSample(signal=signal, subject_id=row.subject_id, recording_id=row.record_id, dataset=row.dataset, modality=row.modality, sampling_rate_hz=fs, start_time_s=0.0, end_time_s=signal.shape[1] / fs, channel_names=names, units=units, provenance={"source_path": row.source_path, "source_format": row.source_format, "source_variant": row.source_variant, "dataset_version": row.dataset_version, "reader": type(self).__name__, "reader_version": "1.0"}, metadata={"continuity": row.continuity, "role_reference": row.role_reference})


def build_ecg_index(dataset: str, root: str | Path, version: str, source_subdir: str) -> Path:
    root = Path(root)
    source = root / source_subdir
    rows: list[RecordIndexRow] = []
    headers = sorted(source.rglob("*.hea"))
    for header in headers:
        try:
            nchan, fs, nsamp, names, units, *_ = _header(header)
        except (OSError, ValueError, IndexError):
            continue
        data = header.with_suffix(".dat") if header.with_suffix(".dat").exists() else header.with_suffix(".mat")
        if not data.exists():
            continue
        rel = header.relative_to(root).as_posix()
        record = header.stem
        rows.append(RecordIndexRow(dataset, version, f"{dataset}:record:{record}", record, None, None, "ECG", names, (fs,) * nchan, units, nsamp, nsamp / fs if nsamp else None, rel, "WFDB" if data.suffix == ".dat" else "WFDB-MAT", "source-native", "source-native; no gap repair", "PhysioNet source-native record identity; no separate subject field asserted", ROLE_REFERENCE, "PASS", None))
    output = root / "metadata" / "records.jsonl"
    write_jsonl(rows, output)
    write_hash(output)
    return output


class PPG_BPReader(_Indexed):
    name, version = "ppg-bp", "5"

    def __getitem__(self, index: int) -> UnifiedSample:
        row = self.rows[index]
        with zipfile.ZipFile(self.root / "incoming" / "ppg-bp-v5.zip") as archive:
            with archive.open(row.source_path) as handle:
                signal = np.loadtxt(handle, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[None, :]
        elif signal.shape[0] == row.n_samples and signal.shape[1] == 1:
            signal = signal.T
        return UnifiedSample(signal=signal, subject_id=row.subject_id, recording_id=row.record_id, dataset=self.name, modality="PPG", sampling_rate_hz=float(row.sampling_rate_hz[0]), start_time_s=0.0, end_time_s=signal.shape[1] / row.sampling_rate_hz[0], channel_names=tuple(row.channel_name), units=tuple(row.unit), provenance={"source_path": row.source_path, "source_format": "text-in-zip", "dataset_version": self.version, "reader": type(self).__name__, "reader_version": "1.0"}, metadata={"continuity": row.continuity, "role_reference": row.role_reference})


def build_ppg_bp_index(root: str | Path) -> Path:
    root = Path(root)
    rows: list[RecordIndexRow] = []
    with zipfile.ZipFile(root / "incoming" / "ppg-bp-v5.zip") as archive:
        for name in sorted(n for n in archive.namelist() if n.lower().endswith(".txt") and "/" in n):
            match = re.search(r"/(\d+)_([123])\.txt$", name)
            if not match:
                continue
            subject, trial = match.groups()
            with archive.open(name) as handle:
                n_samples = sum(1 for line in handle if line.strip())
            rows.append(RecordIndexRow("ppg-bp", "5", f"ppg-bp:{subject}", f"{subject}:{trial}", None, trial, "PPG", ("PPG",), (100.0,), ("UNKNOWN",), n_samples, n_samples / 100.0, name, "text-in-zip", "source-native", "source-native; no gap repair", "PPG-BP source subject and trial identifiers", ROLE_REFERENCE, "PASS", None))
    output = root / "metadata" / "records.jsonl"
    write_jsonl(rows, output)
    write_hash(output)
    return output


WESAD_VARIANTS = (("chest", "ECG", 700.0, ("ECG",)), ("chest", "ACC", 700.0, ("ACC_X", "ACC_Y", "ACC_Z")), ("chest", "EDA", 700.0, ("EDA",)), ("chest", "EMG", 700.0, ("EMG",)), ("chest", "Temp", 700.0, ("Temp",)), ("chest", "Resp", 700.0, ("Resp",)), ("wrist", "BVP", 64.0, ("BVP",)), ("wrist", "ACC", 32.0, ("ACC_X", "ACC_Y", "ACC_Z")), ("wrist", "EDA", 4.0, ("EDA",)), ("wrist", "TEMP", 4.0, ("TEMP",)))


class WESADReader(_Indexed):
    name, version = "wesad", "1"

    def __getitem__(self, index: int) -> UnifiedSample:
        row = self.rows[index]
        with zipfile.ZipFile(self.root / "incoming" / "WESAD.zip") as archive:
            with archive.open(row.source_path) as handle:
                payload = pickle.load(handle, encoding="latin1")
        body, key = row.source_variant.split(":", 1)
        signal = np.asarray(payload["signal"][body][key], dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[None, :]
        else:
            signal = signal.T
        fs = float(row.sampling_rate_hz[0])
        return UnifiedSample(signal=signal, subject_id=row.subject_id, recording_id=row.record_id, dataset=self.name, modality=row.modality, sampling_rate_hz=fs, start_time_s=0.0, end_time_s=signal.shape[1] / fs, channel_names=tuple(row.channel_name), units=tuple(row.unit), provenance={"source_path": row.source_path, "source_format": "pickle-in-zip", "source_variant": row.source_variant, "dataset_version": self.version, "reader": type(self).__name__, "reader_version": "1.0"}, labels={"source_label_key": "label"}, metadata={"continuity": row.continuity, "role_reference": row.role_reference})


def build_wesad_index(root: str | Path) -> Path:
    root = Path(root)
    rows: list[RecordIndexRow] = []
    with zipfile.ZipFile(root / "incoming" / "WESAD.zip") as archive:
        subjects = sorted(n for n in archive.namelist() if re.search(r"WESAD/S\d+/S\d+\.pkl$", n))
    for source in subjects:
        subject = Path(source).stem
        for body, key, fs, channels in WESAD_VARIANTS:
            rows.append(RecordIndexRow("wesad", "1", f"wesad:{subject}", f"{subject}:{body}:{key}", None, None, "multimodal", channels, (fs,) * len(channels), ("UNKNOWN",) * len(channels), None, None, source, "pickle-in-zip", f"{body}:{key}", "source-native; no gap repair", "WESAD source subject ID", ROLE_REFERENCE, "PASS", None))
    output = root / "metadata" / "records.jsonl"
    write_jsonl(rows, output)
    write_hash(output)
    return output


class LUDBReader(ECGReader):
    name, version = "ludb", "1.0.1"


class MITBIHReader(ECGReader):
    name, version = "mit-bih", "1.0.0"


class ChapmanShaoxingReader(ECGReader):
    name, version = "chapman-shaoxing", "1.0.0"


class GeorgiaReader(ECGReader):
    name, version = "georgia", "1.0.2"


class CPSC2018Reader(ECGReader):
    name, version = "cpsc2018", "1.0.2"
