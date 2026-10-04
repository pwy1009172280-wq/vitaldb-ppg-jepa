"""Generate the MIMIC legitimate segment-header required list.

The parser operates on authoritative WFDB multi-segment master headers. Comment
lines are metadata and must never become segment references. Layout and gap
records are recognized for auditability but are not segment-header targets.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


SEGMENT_TOKEN = re.compile(r"^[0-9]+_[0-9]+$")


@dataclass
class ParseSummary:
    masters: int
    raw_references: int
    required: int
    duplicates: int
    comments: int
    layouts: int
    gaps: int
    malformed: int
    malformed_examples: list[tuple[str, int, str]]


def parse_master(master_path: Path, headers_root: Path) -> tuple[list[str], int, int, int, list[tuple[str, int, str]]]:
    lines = master_path.read_text(errors="replace").splitlines()
    malformed: list[tuple[str, int, str]] = []
    refs: list[str] = []
    comments = layouts = gaps = 0

    if not lines or len(lines[0].split()) < 2:
        malformed.append((str(master_path), 1, "bad master declaration"))

    for lineno, raw in enumerate(lines[1:], 2):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments += 1
            continue

        parts = line.split()
        token = parts[0]
        if token.endswith("_layout"):
            layouts += 1
            if len(parts) < 2:
                malformed.append((str(master_path), lineno, line))
            continue
        if token == "~":
            gaps += 1
            if len(parts) < 2:
                malformed.append((str(master_path), lineno, line))
            continue
        if len(parts) >= 2 and SEGMENT_TOKEN.fullmatch(token):
            subject = master_path.relative_to(headers_root).parent
            refs.append(f"{subject}/{token}.hea")
        else:
            malformed.append((str(master_path), lineno, line))

    return refs, comments, layouts, gaps, malformed


def generate_required_list(root: Path) -> tuple[list[str], ParseSummary]:
    headers_root = root / "headers_v2"
    master_list = root / "mimic_biosignal_master_headers.txt"
    masters = [x.strip() for x in master_list.read_text().splitlines() if x.strip()]
    raw: list[str] = []
    comments = layouts = gaps = 0
    malformed: list[tuple[str, int, str]] = []

    for relative in masters:
        refs, c, l, g, bad = parse_master(headers_root / relative, headers_root)
        raw.extend(refs)
        comments += c
        layouts += l
        gaps += g
        malformed.extend(bad)

    required = sorted(set(raw))
    summary = ParseSummary(
        masters=len(masters), raw_references=len(raw), required=len(required),
        duplicates=len(raw) - len(required), comments=comments, layouts=layouts,
        gaps=gaps, malformed=len(malformed), malformed_examples=malformed[:20],
    )
    return required, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()

    required, summary = generate_required_list(args.root)
    args.output.write_text("\n".join(required) + "\n")
    args.summary.write_text(json.dumps(summary.__dict__, indent=2) + "\n")
    print(json.dumps(summary.__dict__, indent=2))
    return 0 if summary.required == 870230 and summary.duplicates == 0 and summary.malformed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
