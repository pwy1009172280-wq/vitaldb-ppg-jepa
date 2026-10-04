"""Generic, evidence-producing readiness checker for Data Bank Phase 1."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from src.data.index import read_jsonl, sha256_file


REQUIRED = (
    "source_verified", "acquisition_complete", "files_readable",
    "subject_identity_available", "recording_identity_available",
    "signal_channels_documented", "sampling_rate_documented",
    "units_documented_or_explicitly_unknown", "continuity_preserved",
    "provenance_recorded", "manifest_complete", "role_policy_recorded",
    "overlap_status_recorded", "canonical_reader_validated",
)


def _status_file_passes(root: Path) -> bool:
    statuses = list(root.glob("manifest/status.*"))
    if not statuses:
        return False
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in statuses)
    return "exit_code=0" in text or "download_exit_code=0" in text


def _manifest_complete(root: Path, index_path: Path) -> bool:
    return index_path.is_file() and index_path.stat().st_size > 0 and _status_file_passes(root)


def check_readiness(root: str | Path, *, dataset: str, reader_smoke: str | Path | None = None) -> dict[str, Any]:
    root = Path(root)
    index_path = root / "metadata" / "records.jsonl"
    provenance = root / "metadata" / "source_provenance.tsv"
    role_policy = root.parent.parent / "registry" / "role_policy.yaml"
    overlap_map = root.parent.parent / "registry" / "overlap_map.yaml"
    rows = list(read_jsonl(index_path)) if index_path.exists() else []
    checks: dict[str, dict[str, Any]] = {}

    def put(name: str, status: str, evidence: str) -> None:
        checks[name] = {"status": status, "evidence": evidence}

    put("source_verified", "PASS" if provenance.exists() else "FAIL", str(provenance))
    put("acquisition_complete", "PASS" if _status_file_passes(root) else "FAIL", str(root / "manifest"))
    put("files_readable", "PASS" if rows else "FAIL", f"{index_path} rows={len(rows)}")
    put("subject_identity_available", "PASS" if rows and all(row.subject_id for row in rows) else "FAIL", str(index_path))
    put("recording_identity_available", "PASS" if rows and all(row.record_id for row in rows) else "FAIL", str(index_path))
    put("signal_channels_documented", "PASS" if rows and all(row.channel_name for row in rows) else "FAIL", str(index_path))
    put("sampling_rate_documented", "PASS" if rows and all(all(rate is not None for rate in row.sampling_rate_hz) for row in rows) else "FAIL", str(index_path))
    put("units_documented_or_explicitly_unknown", "PASS" if rows and all(row.unit for row in rows) else "FAIL", str(index_path))
    put("continuity_preserved", "PASS" if rows and all(row.continuity for row in rows) else "FAIL", str(index_path))
    put("provenance_recorded", "PASS" if rows and provenance.exists() and all(row.provenance for row in rows) else "FAIL", str(provenance))
    put("manifest_complete", "PASS" if _manifest_complete(root, index_path) else "FAIL", str(root / "manifest"))
    put("role_policy_recorded", "PASS" if role_policy.exists() else "FAIL", str(role_policy))
    put("overlap_status_recorded", "PASS" if overlap_map.exists() else "FAIL", str(overlap_map))
    smoke_path = Path(reader_smoke) if reader_smoke else root / "metadata" / "reader_smoke.json"
    smoke_ok = False
    if smoke_path.exists():
        try:
            smoke_ok = json.loads(smoke_path.read_text(encoding="utf-8")).get("status") == "PASS"
        except (OSError, ValueError):
            smoke_ok = False
    put("canonical_reader_validated", "PASS" if smoke_ok else "FAIL", str(smoke_path))
    passed = all(checks[name]["status"] == "PASS" for name in REQUIRED)
    return {
        "dataset": dataset,
        "state": "READY_FOR_PREPROCESS" if passed else "NOT_READY",
        "checks": checks,
        "index_sha256": sha256_file(index_path) if index_path.exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_readiness(args.root, dataset=args.dataset)
    output = args.output or args.root / "metadata" / "readiness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["state"] == "READY_FOR_PREPROCESS" else 1)


if __name__ == "__main__":
    main()
