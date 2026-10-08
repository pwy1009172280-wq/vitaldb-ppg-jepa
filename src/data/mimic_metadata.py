"""Validated MIMIC logical-path and channel helpers (no WFDB grammar here).

Reuses the required-list parser's path/segment rules. Official WFDB header/signal
semantics (signal name, format, gain, baseline, units, master/layout/segment
interpretation) remain the authority in the native reader; this module only
turns a logical header path into group/subject/segment and applies the explicit
channel selection rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SEGMENT_TOKEN = re.compile(r"^[0-9]+_[0-9]+$")
GROUP_TOKEN = re.compile(r"^p[0-9]{2}$")
SUBJECT_TOKEN = re.compile(r"^p[0-9]{6}$")

# Directory components that are never MIMIC subjects.
FORBIDDEN_SUBJECT_COMPONENTS = frozenset({"root", "queue", "headers_v2", "archives"})


@dataclass(frozen=True)
class MimicLogicalRef:
    group: str | None
    subject: str
    segment: str
    logical_path: str


def parse_subject_component(component: str) -> str | None:
    """Return a canonical MIMIC subject ID for a path component, or None.

    Only the 6-digit ``pNNNNNN`` directory is a subject. Group directories
    (``p00``), root, and queue are explicitly rejected.
    """
    if component in FORBIDDEN_SUBJECT_COMPONENTS:
        return None
    if not SUBJECT_TOKEN.fullmatch(component):
        return None
    return component


def parse_logical_path(logical_path: str) -> MimicLogicalRef:
    """Parse a MIMIC logical header path into group/subject/segment.

    The authoritative required-list shape is ``<group>/<subject>/<segment>.hea``.
    """
    path = Path(logical_path)
    if path.suffix != ".hea":
        raise ValueError(f"MIMIC logical path must end in .hea: {logical_path!r}")
    parts = path.parts
    if len(parts) < 2:
        raise ValueError(f"MIMIC logical path has too few components: {logical_path!r}")
    segment = path.stem
    if not SEGMENT_TOKEN.fullmatch(segment):
        raise ValueError(f"MIMIC segment token is not NUMBER_NUMBER: {segment!r}")
    subject = parse_subject_component(parts[-2])
    if subject is None:
        raise ValueError(
            f"MIMIC subject component is not a 6-digit pNNNNNN: {parts[-2]!r}"
        )
    group = parts[-3] if len(parts) >= 3 and GROUP_TOKEN.fullmatch(parts[-3]) else None
    return MimicLogicalRef(
        group=group, subject=subject, segment=segment, logical_path=logical_path
    )


def select_channel_indices(sig_names: tuple[str, ...], source_label: str) -> list[int]:
    """Return indices of decoded channels whose name matches ``source_label``.

    The channel selection rule is configuration; the decoded ``sig_name`` must
    come from the official WFDB header, never from a hand-rolled parser.
    """
    if not source_label:
        raise ValueError("channel selection source_label must be non-empty")
    wanted = source_label.upper()
    return [index for index, name in enumerate(sig_names) if str(name).upper() == wanted]
