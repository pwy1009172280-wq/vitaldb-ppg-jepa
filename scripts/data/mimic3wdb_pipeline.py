#!/usr/bin/env python3
"""Inventory, plan, download, and process MIMIC-III WDB Matched PPG.

The default commands are read-only (``inventory`` and ``plan``).  This module
deliberately does not know about VitalDB and never joins independent WFDB
segments.  A manifest row represents one PLETH signal in one physical segment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_SIGNALS = ("PLETH",)


@dataclass(frozen=True)
class PipelineConfig:
    signal_names: tuple[str, ...] = DEFAULT_SIGNALS
    resample_hz: float | None = None
    window_seconds: float | None = None
    normalization: str = "none"
    filter_name: str = "none"
    delete_raw_requires_sha256: bool = True


@dataclass
class Header:
    path: str
    record_name: str
    n_sig: int
    sampling_rate: float
    duration_seconds: float | None
    signals: list[dict]
    segments: list[dict]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_header(path: Path) -> Header:
    """Parse the stable, textual portion of a WFDB header.

    Signal and segment rows are kept separately.  The parser is intentionally
    conservative: fields it cannot interpret remain in ``raw_tokens`` rather
    than being guessed.
    """
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines()]
    content = [line for line in lines if line and not line.startswith("#")]
    if not content:
        raise ValueError(f"empty WFDB header: {path}")
    first = content[0].split()
    if len(first) < 3:
        raise ValueError(f"invalid WFDB record line: {path}")
    try:
        n_sig = int(first[1])
        fs = float(re.split(r"[()/]", first[2])[0])
    except ValueError as exc:
        raise ValueError(f"invalid WFDB record metadata: {path}") from exc
    duration = None
    if len(first) >= 4:
        duration = _float(first[3])
        if duration is not None and fs:
            duration = duration / fs if duration > 100 else duration
    signals = []
    for index, line in enumerate(content[1 : 1 + n_sig]):
        tokens = line.split()
        # WFDB: file, fmt, gain, baseline, units, adc_res, adc_zero,
        # initial_value, checksum, block_size, signal_name.
        signals.append({
            "signal_index": index,
            "file": tokens[0] if tokens else "",
            "format": tokens[1] if len(tokens) > 1 else "",
            "gain": _float(tokens[2]) if len(tokens) > 2 else None,
            "baseline": tokens[3] if len(tokens) > 3 else "",
            "units": tokens[4] if len(tokens) > 4 else "",
            # Some WDB headers combine gain/units (name at token 9), while
            # canonical WFDB headers keep a separate block-size field (name
            # at token 10).  Prefer the latter only when it is non-numeric.
            "signal_name": (tokens[10] if len(tokens) > 10 and _float(tokens[10]) is None else (tokens[9] if len(tokens) > 9 else (tokens[-1] if len(tokens) > 5 else ""))),
            "raw_tokens": tokens,
        })
    segments = []
    for index, line in enumerate(content[1 + n_sig :]):
        tokens = line.split()
        if len(tokens) >= 2 and _float(tokens[1]) is not None:
            segments.append({
                "segment_index": index,
                "record_name": tokens[0],
                "sample_count": int(float(tokens[1])),
                "raw_tokens": tokens,
            })
    return Header(str(path), first[0], n_sig, fs, duration, signals, segments)


def _subject_id(path: Path, record_name: str, pattern: str | None) -> str:
    if pattern:
        match = re.search(pattern, str(path))
        if match:
            return match.group(1) if match.groups() else match.group(0)
    # Do not infer a patient identifier from a record identifier by default.
    return ""


def _norm(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def build_manifest(
    headers_root: Path,
    *,
    signal_names: tuple[str, ...] = DEFAULT_SIGNALS,
    subject_regex: str | None = None,
    source_base_url: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return PLETH rows, required files, and parse failures without I/O writes."""
    rows, required, failures = [], [], []
    wanted = {_norm(name) for name in signal_names}
    for header_path in sorted(headers_root.rglob("*.hea")):
        try:
            header = parse_header(header_path)
        except (OSError, ValueError) as exc:
            failures.append({"path": str(header_path), "stage": "parse_header", "error": str(exc)})
            continue
        subject = _subject_id(header_path, header.record_name, subject_regex)
        selected = [s for s in header.signals if _norm(s["signal_name"]) in wanted]
        if not selected:
            continue
        # A single-segment header is itself the physical segment.  A
        # multi-segment layout header contributes one row per listed segment.
        segments = header.segments or [{"segment_index": 0, "record_name": header.record_name, "sample_count": ""}]
        for segment in segments:
            segment_name = str(segment["record_name"])
            segment_header = (header_path.parent / (segment_name if segment_name.endswith(".hea") else segment_name + ".hea")).resolve()
            for signal in selected:
                data_name = signal["file"]
                data_path = (segment_header.parent / data_name).resolve() if data_name else None
                row = {
                    "record_id": header.record_name,
                    "subject_id": subject,
                    "layout_header_path": str(header_path.resolve()),
                    "segment_index": segment["segment_index"],
                    "segment_record_id": segment_name,
                    "segment_header_path": str(segment_header),
                    "data_path": str(data_path) if data_path else "",
                    "signal_index": signal["signal_index"],
                    "signal_name": signal["signal_name"],
                    "format": signal["format"],
                    "sampling_rate": header.sampling_rate,
                    "gain": signal["gain"],
                    "baseline": signal["baseline"],
                    "units": signal["units"],
                    "sample_count": segment["sample_count"],
                    "continuity_status": "segment_boundary_preserved",
                    "source_base_url": source_base_url or "",
                    "status": "inventoried",
                }
                rows.append(row)
                for kind, path in (("segment_header", segment_header), ("data", data_path)):
                    if path:
                        relative = str(path.relative_to(headers_root.resolve())) if path.is_relative_to(headers_root.resolve()) else str(path)
                        required.append({"record_id": header.record_name, "segment_record_id": segment_name, "kind": kind, "path": str(path), "relative_path": relative})
                        if source_base_url and not Path(relative).is_absolute():
                            required[-1]["source_url"] = urljoin(source_base_url.rstrip("/") + "/", relative)
    return rows, required, failures


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def atomic_download(url: str, destination: Path, retries: int = 3, timeout: int = 120) -> str:
    """Download with atomic finalization and best-effort HTTP range resume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return "skipped_existing"
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        request = Request(url, headers={"User-Agent": "vitaldb-ppg-jepa/1.0"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urlopen(request, timeout=timeout) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                if mode == "wb":
                    offset = 0
                with partial.open(mode) as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            os.replace(partial, destination)
            return "downloaded" if not offset else "resumed"
        except (HTTPError, URLError, TimeoutError, OSError):
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def validate_processed_output(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"missing/empty processed output: {path}")
    with npz_reader(path) as archive:
        required = {"signal", "metadata_json"}
        if not required.issubset(set(archive.files)):
            raise ValueError(f"processed output lacks {required}: {path}")
        signal = archive["signal"]
        if signal.ndim != 1 or signal.size == 0:
            raise ValueError(f"invalid processed signal: {path}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        source = Path(metadata["source_data_path"])
        if source.exists() and metadata.get("source_sha256") != sha256(source):
            raise ValueError(f"source hash mismatch: {path}")
        if not metadata.get("source_sha256"):
            raise ValueError(f"processed output lacks source hash: {path}")
        return metadata


class npz_reader:
    """Small import-local wrapper so inventory does not require NumPy."""
    def __init__(self, path: Path):
        self.path = path
    def __enter__(self):
        import numpy as np
        self.archive = np.load(self.path, allow_pickle=False)
        return self.archive
    def __exit__(self, *_):
        self.archive.close()


def safe_delete_raw(raw_path: Path, processed_path: Path, raw_root: Path) -> None:
    """Delete one raw file only after a validated output proves its identity."""
    raw = raw_path.resolve()
    root = raw_root.resolve()
    if not raw.is_relative_to(root):
        raise ValueError(f"refusing to delete outside raw root: {raw}")
    metadata = validate_processed_output(processed_path)
    if Path(metadata["source_data_path"]).resolve() != raw:
        raise ValueError("processed provenance does not identify requested raw file")
    if not raw.exists():
        raise FileNotFoundError(raw)
    if metadata["source_sha256"] != sha256(raw):
        raise ValueError("raw file hash changed before deletion")
    raw.unlink()


def process_row(row: dict, output_root: Path, *, delete_raw: bool = False, raw_root: Path | None = None) -> dict:
    """Process one segment without joining it to another segment.

    Filtering, resampling, normalization, and windowing are intentionally not
    performed here.  Those choices are config decisions, not hidden defaults.
    Requires the optional ``wfdb`` package only when this command is run.
    """
    import numpy as np
    import wfdb
    data_path = Path(row["data_path"])
    record = wfdb.rdrecord(str(data_path.with_suffix("")), channels=[int(row["signal_index"])])
    signal = np.asarray(record.p_signal[:, 0] if record.p_signal is not None else record.d_signal[:, 0], dtype=np.float32)
    output = output_root / f"{row['record_id']}__segment_{int(row['segment_index']):04d}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(row)
    metadata.update({"source_data_path": str(data_path.resolve()), "source_sha256": sha256(data_path), "processing": {"filter": "none", "resample_hz": None, "normalization": "none", "window_seconds": None}})
    tmp = output.with_suffix(output.suffix + ".part")
    # Passing an open handle prevents NumPy from appending a second .npz
    # suffix to the temporary filename.
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, signal=signal, metadata_json=json.dumps(metadata, sort_keys=True))
    os.replace(tmp, output)
    validate_processed_output(output)
    if delete_raw:
        if raw_root is None:
            raise ValueError("--delete-raw requires --raw-root")
        safe_delete_raw(data_path, output, raw_root)
    return {**row, "status": "processed", "processed_path": str(output), "source_sha256": metadata["source_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="scan local headers; no downloads")
    inv.add_argument("--headers-root", type=Path, required=True)
    inv.add_argument("--output-root", type=Path, required=True)
    inv.add_argument("--subject-regex")
    inv.add_argument("--source-base-url")
    inv.add_argument("--signal", action="append", default=list(DEFAULT_SIGNALS))
    plan = sub.add_parser("plan", help="show exact required files; no downloads")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    dl = sub.add_parser("download", help="download only files listed by a plan; resumable")
    dl.add_argument("--plan", type=Path, required=True)
    dl.add_argument("--raw-root", type=Path, required=True)
    dl.add_argument("--source-base-url", required=True)
    dl.add_argument("--limit", type=int)
    proc = sub.add_parser("process", help="process completed manifest rows one segment at a time")
    proc.add_argument("--manifest", type=Path, required=True)
    proc.add_argument("--output-root", type=Path, required=True)
    proc.add_argument("--processed-manifest", type=Path, required=True)
    proc.add_argument("--delete-raw", action="store_true")
    proc.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    if args.command == "inventory":
        rows, required, failures = build_manifest(args.headers_root, signal_names=tuple(args.signal), subject_regex=args.subject_regex, source_base_url=args.source_base_url)
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_root / "ppg_manifest.csv", rows)
        write_csv(args.output_root / "required_files.csv", required)
        write_csv(args.output_root / "failures.csv", failures)
        print(json.dumps({"pleth_rows": len(rows), "required_files": len(required), "failures": len(failures)}, indent=2))
    elif args.command == "plan":
        with args.manifest.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = []
        for row in rows:
            for kind, key in (("segment_header", "segment_header_path"), ("data", "data_path")):
                if row.get(key):
                    required.append({"record_id": row["record_id"], "segment_record_id": row["segment_record_id"], "kind": kind, "path": row[key], "relative_path": Path(row[key]).name, "status": "existing" if Path(row[key]).exists() else "missing"})
        write_csv(args.output, required)
        print(json.dumps({"required_files": len(required), "missing": sum(r["status"] == "missing" for r in required)}, indent=2))
    elif args.command == "download":
        with args.plan.open(newline="") as stream:
            entries = list(csv.DictReader(stream))
        results = []
        for entry in entries[: args.limit]:
            relative = entry.get("relative_path") or Path(entry["path"]).name
            destination = args.raw_root / relative
            source = entry.get("source_url") or urljoin(args.source_base_url.rstrip("/") + "/", relative)
            try:
                status = atomic_download(source, destination)
                result = {**entry, "destination": str(destination), "source_url": source, "download_status": status}
                print(f"{status}: {destination}")
            except Exception as exc:  # keep the batch restartable
                result = {**entry, "destination": str(destination), "source_url": source, "download_status": "failed", "error": repr(exc)}
                print(f"failed: {destination}: {exc}")
            results.append(result)
            write_csv(args.plan.with_name(args.plan.stem + "_downloaded.csv"), results)
        write_csv(args.plan.with_name(args.plan.stem + "_downloaded.csv"), results)
    elif args.command == "process":
        if args.delete_raw and args.raw_root is None:
            parser.error("process --delete-raw requires --raw-root")
        with args.manifest.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        results = []
        for row in rows:
            output = args.output_root / f"{row['record_id']}__segment_{int(row['segment_index']):04d}.npz"
            if output.exists():
                try:
                    validate_processed_output(output)
                    results.append({**row, "status": "skipped_validated", "processed_path": str(output)})
                    continue
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    output.unlink()
            try:
                results.append(process_row(row, args.output_root, delete_raw=args.delete_raw, raw_root=args.raw_root))
            except Exception as exc:
                results.append({**row, "status": "failed", "error": repr(exc)})
            write_csv(args.processed_manifest, results)
        write_csv(args.processed_manifest, results)
        print(json.dumps({"processed": sum(r["status"] == "processed" for r in results), "skipped": sum(r["status"] == "skipped_validated" for r in results)}, indent=2))


if __name__ == "__main__":
    main()
