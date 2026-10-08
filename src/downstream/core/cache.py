"""Strict, content-addressed feature cache metadata."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .extraction import ExtractedFeature


CACHE_SCHEMA_VERSION = 1


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def checkpoint_content_hash(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compute_sample_set_hash(samples: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> str:
    return _digest(list(samples))


@dataclass(frozen=True)
class FeatureCacheKey:
    dataset_manifest_hash: str
    split_sample_set_hash: str
    preprocessing_version_hash: str
    encoder_checkpoint_hash: str
    reader_name: str
    reader_version: str
    representation_name: str
    pooling_name: str
    pooling_version: str
    feature_dtype: str = "float32"
    extraction_schema_version: int = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.extraction_schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported feature cache schema version")
        if self.feature_dtype != "float32":
            raise ValueError("v1 feature dtype must be float32")
        if any(not isinstance(value, str) or not value for value in asdict(self).values() if isinstance(value, str)):
            raise ValueError("cache key fields must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_feature_cache(feature: ExtractedFeature, path: str | Path, key: FeatureCacheKey, *, git_commit: str | None = None) -> Path:
    if feature.representation != key.representation_name:
        raise ValueError("feature representation does not match cache key")
    if feature.reader_name != key.reader_name or feature.reader_version != key.reader_version:
        raise ValueError("feature reader does not match cache key")
    if feature.pooling != key.pooling_name:
        raise ValueError("feature pooling does not match cache key")
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": key.to_dict(),
        "git_commit": git_commit,
        "split": feature.split,
        "checkpoint_reference": feature.checkpoint_reference,
        "samples": feature.samples,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=feature.values,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        )
    return output


def load_feature_cache(path: str | Path, expected_key: FeatureCacheKey) -> ExtractedFeature:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("unknown feature cache schema version")
        if metadata.get("cache_key") != expected_key.to_dict():
            raise ValueError("feature cache key mismatch; cache miss")
        key = metadata["cache_key"]
        values = np.asarray(archive["features"], dtype=np.float32)
    return ExtractedFeature(
        key["representation_name"], values, tuple(metadata["samples"]), metadata["split"],
        key["reader_name"], key["reader_version"], key["pooling_name"], metadata["checkpoint_reference"],
    )
