"""Versioned downstream result serialization."""

import json
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = 1


def save_results(payload: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = dict(payload)
    content["schema_version"] = RESULT_SCHEMA_VERSION
    output.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return output


def load_results(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != RESULT_SCHEMA_VERSION:
        raise ValueError("unknown result schema version")
    return payload
