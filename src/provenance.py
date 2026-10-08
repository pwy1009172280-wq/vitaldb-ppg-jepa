"""Canonical serialization and hashing — Serialization v1.

Single low-dependency authority for turning pipeline objects (configs, sample
identities, metadata) into stable, comparable canonical JSON and content
digests. Every other identity/hash helper must delegate here instead of
re-implementing ``json.dumps`` rules.

Canonical float encoding (MINOR-4) — the one locked implementation is the
CPython :mod:`json` float serializer (``float.__repr__`` shortest round-trip)
with ``allow_nan=False``:

* ``1`` (int) -> ``1``; ``1.0`` (float) -> ``1.0``  (float vs int stays distinct)
* ``-0.0`` -> ``-0.0``                              (negative zero preserved)
* ``0.1`` -> ``0.1``                                (shortest round-trip)
* NaN / +Inf / -Inf raise ``ValueError``            (fail closed)

Rules (plan "Serialization v1"):
  dict/Mapping  -> str keys only, sorted by Unicode codepoint, recursive.
  dataclass     -> declared fields only (no __dict__).
  list/tuple    -> ordered list (order semantic).
  set/frozenset -> rejected; caller must sort deterministically.
  None/bool/int -> JSON null/true/false/integer; bool is never coerced to int.
  str           -> UTF-8, no BOM, no Unicode normalization, newlines preserved.
  Path          -> rejected (locator, not content).
  NumPy scalar  -> bool/int/finite-float only.
  NumPy array   -> typed descriptor {dtype, shape, C-order LE bytes sha256};
                   object/unicode/unsupported dtype and non-finite float
                   arrays are rejected.

Exposed primitives:
  canonical_json(obj)   -> str  (canonical JSON document)
  content_hash(kind,obj)-> str  (sha256 of a versioned, kind-scoped envelope)
  file_sha256(path)     -> str  (exact bytes; for checkpoints/payloads)
A bare locator string is a *reference*, not content; consume it only after
verifying the expected content_hash/file_sha256 of what it points to.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

CANONICAL_VERSION = 1

_KIND_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Reserved canonical tag; genuine ndarray descriptors are produced only here
# and user mappings must not contain this key.
_NDARRAY_TAG = "__canonical_ndarray__"

_SUPPORTED_NDARRAY_KINDS = frozenset("biuf")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _normalize_float(value: float) -> float:
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite float is not canonical: {value!r}")
    return value


def _normalize_ndarray(arr: Any) -> dict[str, Any]:
    import numpy as np

    if arr.dtype.kind not in _SUPPORTED_NDARRAY_KINDS:
        raise TypeError(f"unsupported ndarray dtype for canonical identity: {arr.dtype}")
    if arr.dtype.kind == "f" and not bool(np.isfinite(arr).all()):
        raise ValueError("non-finite ndarray values are not canonical")
    little = arr.dtype.newbyteorder("<")
    contiguous = np.ascontiguousarray(arr, dtype=little)
    return {
        _NDARRAY_TAG: {
            "dtype": str(little),
            "shape": list(arr.shape),
            "bytes_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    }


def _normalize_numpy_scalar(value: Any) -> Any:
    item = value.item()
    if isinstance(item, bool):
        return bool(item)
    if isinstance(item, int):
        return int(item)
    if isinstance(item, float):
        return _normalize_float(float(item))
    raise TypeError(f"unsupported numpy scalar dtype: {value.dtype}")


def _normalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return _normalize_float(obj)
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"canonical mapping keys must be str, got {type(key).__name__}")
            if key == _NDARRAY_TAG:
                raise TypeError(f"reserved canonical key {_NDARRAY_TAG!r} must not appear in user mappings")
            out[key] = _normalize(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_normalize(item) for item in obj]
    if isinstance(obj, (set, frozenset)):
        raise TypeError("set/frozenset are not canonical; sort deterministically to a list first")
    if isinstance(obj, Path):
        raise TypeError("Path is a locator, not canonical content; resolve to a logical reference first")
    if is_dataclass(obj) and not isinstance(obj, type):
        return _normalize({field.name: getattr(obj, field.name) for field in fields(obj)})
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        if isinstance(obj, np.generic):
            return _normalize_numpy_scalar(obj)
        if isinstance(obj, np.ndarray):
            return _normalize_ndarray(obj)
    raise TypeError(f"unsupported canonical type: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Return the canonical JSON document for ``obj`` (Serialization v1)."""
    return _json_dumps(_normalize(obj))


def content_hash(kind: str, obj: Any) -> str:
    """Return a kind-scoped, versioned canonical content digest.

    ``kind`` is a non-empty token matching ``[A-Za-z0-9._-]+``. Two different
    kinds never collide even for identical payloads.
    """
    if not isinstance(kind, str) or not _KIND_RE.match(kind):
        raise ValueError(f"kind must be a non-empty str matching {_KIND_RE.pattern!r}")
    payload = canonical_json(obj)
    envelope = (
        '{"kind":' + json.dumps(kind) + ',"canonical_version":' + str(CANONICAL_VERSION)
        + ',"payload":' + payload + "}"
    )
    return hashlib.sha256(envelope.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the exact byte SHA256 of a file (streaming, 1 MiB chunks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
