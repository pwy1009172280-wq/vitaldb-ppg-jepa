"""Serialization and reproducibility helpers for experiment runs."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import os
import random
import platform
import sys
from typing import Any, Mapping

import yaml


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def serialize_config(config: Any) -> dict[str, Any]:
    """Convert a config object into a plain YAML/JSON-compatible mapping."""
    serialized = _to_serializable(config)
    if not isinstance(serialized, dict):
        raise TypeError("config must serialize to a mapping")
    return serialized


def save_config(config: Any, path: str | Path) -> Path:
    """Write a resolved configuration as stable, human-readable YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serialize_config(config), handle, sort_keys=False)
    return output


def save_json_metadata(metadata: Any, path: str | Path) -> Path:
    """Write metadata as indented JSON with a trailing newline."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def collect_environment_info() -> dict[str, str]:
    """Collect lightweight environment metadata without scanning datasets."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed Python and available numerical backends without requiring them.

    Optional backends are intentionally discovered at runtime so metadata-only
    infrastructure remains usable in lightweight environments.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    seeded = {"python": True, "numpy": False, "torch": False}
    try:
        import numpy as np

        np.random.seed(seed)
        seeded["numpy"] = True
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded["torch"] = True
    except ImportError:
        pass
    return {"seed": seed, "backends": seeded}
