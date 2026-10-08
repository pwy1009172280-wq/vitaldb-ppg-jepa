"""Bounded, read-only source access for archive and loose source members.

No download transport. Safety-critical archive access: reject partial/growing
sources, normalize and validate member paths (no absolute/``..`` escape), reject
symlink/hardlink/device members, reject duplicate member names, and bound
materialization to a run-local scratch directory. Full WFDB closure construction
lives in the native reader on top of these primitives.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PARTIAL_SUFFIXES = (".part", ".partial")


@dataclass(frozen=True)
class MaterializedMember:
    relative_path: str
    bytes_written: int


def is_completed_source(path: str | Path) -> bool:
    """A source is usable only if it exists, is a file, and is not a partial."""
    p = Path(path)
    return p.is_file() and not p.name.endswith(PARTIAL_SUFFIXES)


def normalize_member_name(name: str) -> str:
    """Normalize separators and reject absolute/parent-escape member paths."""
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        raise ValueError(f"unsafe archive member path (absolute): {name!r}")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError(f"unsafe archive member path (parent escape): {name!r}")
    if not parts:
        raise ValueError(f"unsafe archive member path (empty): {name!r}")
    return "/".join(parts)


def _validate_member(info: tarfile.TarInfo) -> str:
    name = normalize_member_name(info.name)
    if info.issym() or info.islnk():
        raise ValueError(f"unsafe archive member (symlink/hardlink): {name!r}")
    if not (info.isfile() or info.isdir()):
        raise ValueError(f"unsafe archive member type {info.type!r}: {name!r}")
    return name


def iter_regular_members(archive_path: str | Path) -> Iterable[str]:
    """Yield normalized names of regular-file members, validating safety."""
    if not is_completed_source(archive_path):
        raise ValueError(f"source is partial or missing: {archive_path}")
    with tarfile.open(archive_path) as archive:
        for info in archive.getmembers():
            name = _validate_member(info)
            if info.isfile():
                yield name


def materialize_members(
    archive_path: str | Path,
    member_names: Iterable[str],
    dest_dir: str | Path,
    *,
    max_bytes: int,
) -> list[MaterializedMember]:
    """Extract a bounded set of validated members into ``dest_dir``.

    Each member is written only after its normalized name passes validation, so
    extraction can never escape ``dest_dir``. Duplicate member names fail closed.
    """
    if not is_completed_source(archive_path):
        raise ValueError(f"source is partial or missing: {archive_path}")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    seen: set[str] = set()
    materialized: list[MaterializedMember] = []
    with tarfile.open(archive_path) as archive:
        for member in member_names:
            info = archive.getmember(member)
            name = _validate_member(info)
            if not info.isfile():
                raise ValueError(f"member is not a regular file: {name!r}")
            if name in seen:
                raise ValueError(f"duplicate archive member: {name!r}")
            seen.add(name)
            if total + info.size > max_bytes:
                raise ValueError(
                    f"materialization would exceed byte cap {max_bytes} "
                    f"(already {total}, adding {info.size})"
                )
            source = archive.extractfile(info)
            if source is None:
                raise ValueError(f"cannot extract member: {name!r}")
            data = source.read()
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            total += len(data)
            materialized.append(MaterializedMember(name, len(data)))
    return materialized
