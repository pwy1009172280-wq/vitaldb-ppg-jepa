#!/usr/bin/env python3
"""Build a metadata-only inventory from local MIMIC WFDB segment headers.

This command reads ``.hea`` files only.  It never opens waveform files and
never changes the input header tree.  One output row represents one physical
header/segment, including headers that do not contain PLETH; this makes the
channel-presence counts auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


_SIGNAL_ALIASES = {
    "pleth": {"PLETH"},
    "ecg": {"ECG", "II", "I", "III", "AVR", "AVL", "AVF", "V"},
    "abp": {"ABP", "ART", "ARTERIAL", "IBP"},
}


@dataclass(frozen=True)
class ParsedHeader:
    header_path: Path
    record_id: str
    n_channels: int
    sampling_frequency: float
    sample_count: int | None
    duration_seconds: float | None
    signal_names: tuple[str, ...]
    signal_metadata: tuple[dict, ...]
    comments: tuple[str, ...]
    first_line: str


def _number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _signal_name(tokens: list[str]) -> str:
    """Extract a WFDB signal name across common header variants."""
    # Canonical WFDB signal rows put the name after block_size (token 9).
    # Some MIMIC/WDB variants include an additional description/name token.
    candidates = []
    if len(tokens) > 10:
        candidates.append(tokens[10])
    if len(tokens) > 9:
        candidates.append(tokens[9])
    if len(tokens) > 8:
        candidates.append(tokens[8])
    candidates.extend(reversed(tokens[5:]))
    for candidate in candidates:
        if candidate and _number(candidate) is None and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", candidate):
            return candidate
    return tokens[-1] if tokens else ""


def parse_header(path: Path) -> ParsedHeader:
    """Parse textual WFDB header metadata without touching waveform files."""
    raw_lines = path.read_text(errors="replace").splitlines()
    if not raw_lines:
        raise ValueError("empty header")
    first_line = ""
    first_index = None
    comments: list[str] = []
    for index, raw in enumerate(raw_lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line[1:].strip())
        elif first_index is None:
            first_line = line
            first_index = index
    if first_index is None:
        raise ValueError("header contains no record line")
    first = first_line.split()
    if len(first) < 3:
        raise ValueError("invalid WFDB record line")
    try:
        n_channels = int(first[1])
        sampling_frequency = float(re.split(r"[()/]", first[2])[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid WFDB record metadata") from exc
    if n_channels < 0 or sampling_frequency <= 0:
        raise ValueError("invalid channel count or sampling frequency")

    sample_count = None
    if len(first) >= 4:
        parsed_count = _number(first[3])
        if parsed_count is not None:
            sample_count = int(parsed_count)

    signal_metadata: list[dict] = []
    signal_names: list[str] = []
    content = [line.strip() for line in raw_lines[first_index + 1 :] if line.strip() and not line.lstrip().startswith("#")]
    for signal_index, line in enumerate(content[:n_channels]):
        tokens = line.split()
        name = _signal_name(tokens)
        signal_names.append(name)
        signal_metadata.append({
            "signal_index": signal_index,
            "file": tokens[0] if tokens else "",
            "format": tokens[1] if len(tokens) > 1 else "",
            "gain": tokens[2] if len(tokens) > 2 else "",
            "baseline": tokens[3] if len(tokens) > 3 else "",
            "units": tokens[4] if len(tokens) > 4 else "",
            "signal_name": name,
            "raw_tokens": tokens,
        })
    if len(signal_metadata) != n_channels:
        raise ValueError(f"expected {n_channels} signal rows, found {len(signal_metadata)}")

    duration_seconds = sample_count / sampling_frequency if sample_count is not None else None
    return ParsedHeader(
        header_path=path,
        record_id=first[0],
        n_channels=n_channels,
        sampling_frequency=sampling_frequency,
        sample_count=sample_count,
        duration_seconds=duration_seconds,
        signal_names=tuple(signal_names),
        signal_metadata=tuple(signal_metadata),
        comments=tuple(comments),
        first_line=first_line,
    )


def _subject_id(relative_path: Path, record_id: str, pattern: str | None) -> str:
    if pattern:
        match = re.search(pattern, str(relative_path))
        if match:
            return match.group(1) if match.groups() else match.group(0)
    for component in relative_path.parts:
        if re.fullmatch(r"p\d+", component, flags=re.IGNORECASE):
            return component
    return ""


def _has_signal(names: tuple[str, ...], kind: str) -> bool:
    wanted = _SIGNAL_ALIASES[kind]
    normalized = {re.sub(r"[^A-Z0-9]", "", name.upper()) for name in names}
    return bool(normalized & wanted)


def inventory_headers(headers_root: Path, subject_regex: str | None = None) -> tuple[list[dict], list[dict]]:
    """Return inventory rows and parse failures, without writing anything."""
    root = headers_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"headers root is not a directory: {root}")
    rows: list[dict] = []
    failures: list[dict] = []
    for path in sorted(root.rglob("*.hea")):
        relative = path.relative_to(root)
        try:
            header = parse_header(path)
        except (OSError, ValueError) as exc:
            failures.append({"header_path": str(path), "relative_path": str(relative), "error": str(exc)})
            continue
        rows.append({
            "subject_id": _subject_id(relative, header.record_id, subject_regex),
            "record_id": header.record_id,
            "segment_id": path.stem,
            "segment_path": str(relative.with_suffix("")),
            "header_path": str(relative),
            "signal_names": json.dumps(list(header.signal_names), separators=(",", ":")),
            "n_channels": header.n_channels,
            "sampling_frequency": header.sampling_frequency,
            "sample_count": header.sample_count if header.sample_count is not None else "",
            "duration_seconds": header.duration_seconds if header.duration_seconds is not None else "",
            "has_pleth": int(_has_signal(header.signal_names, "pleth")),
            "has_ecg": int(_has_signal(header.signal_names, "ecg")),
            "has_abp": int(_has_signal(header.signal_names, "abp")),
            "segment_metadata": json.dumps({
                "first_line": header.first_line,
                "comments": list(header.comments),
                "signals": list(header.signal_metadata),
            }, sort_keys=True, separators=(",", ":")),
            "status": "parsed",
        })
    return rows, failures


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


FIELDNAMES = [
    "subject_id", "record_id", "segment_id", "segment_path", "header_path",
    "signal_names", "n_channels", "sampling_frequency", "sample_count",
    "duration_seconds", "has_pleth", "has_ecg", "has_abp", "segment_metadata", "status",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures-output", type=Path)
    parser.add_argument("--subject-regex")
    args = parser.parse_args()
    rows, failures = inventory_headers(args.headers_root, args.subject_regex)
    write_csv(args.output, FIELDNAMES, rows)
    failures_output = args.failures_output or args.output.with_name(args.output.stem + "_failures.csv")
    write_csv(failures_output, ["header_path", "relative_path", "error"], failures)
    print(json.dumps({
        "headers_root": str(args.headers_root.resolve()),
        "output": str(args.output.resolve()),
        "rows": len(rows),
        "pleth_segments": sum(row["has_pleth"] for row in rows),
        "ecg_segments": sum(row["has_ecg"] for row in rows),
        "abp_segments": sum(row["has_abp"] for row in rows),
        "parse_failures": len(failures),
        "failures_output": str(failures_output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
