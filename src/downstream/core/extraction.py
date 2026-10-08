"""Model-agnostic frozen representation extraction."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...data.access import SplitContext
from ...data.base import SubjectLeakageError
from ...data.samples import UnifiedSample
from .encoder import Representation, freeze_encoder


@dataclass(frozen=True)
class FeatureBatch:
    signal: torch.Tensor
    samples: tuple[UnifiedSample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.signal, torch.Tensor):
            raise TypeError("signal must be a torch.Tensor")
        if not self.samples or any(not isinstance(sample, UnifiedSample) for sample in self.samples):
            raise TypeError("FeatureBatch requires UnifiedSample provenance")
        if self.signal.ndim != 3 or self.signal.shape[0] != len(self.samples):
            raise ValueError("FeatureBatch signal must be [batch, channels, time]")


def collate_feature_samples(samples: Sequence[UnifiedSample]) -> FeatureBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample batch")
    shapes = {np.asarray(sample.signal).shape for sample in samples}
    if len(shapes) != 1:
        raise ValueError("sample signal shapes must agree before extraction")
    signal = torch.from_numpy(np.stack([sample.signal for sample in samples]).astype(np.float32))
    return FeatureBatch(signal, tuple(samples))


def _json_default(value: Any):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"non-serializable provenance: {type(value).__name__}")


def sample_reference(sample: UnifiedSample) -> dict[str, Any]:
    fields = (
        "subject_id", "recording_id", "dataset", "modality", "sampling_rate_hz",
        "start_time_s", "end_time_s", "channel_names", "units", "window_id",
        "window_start_sample", "window_end_sample", "provenance", "metadata", "labels",
    )
    return json.loads(json.dumps({key: getattr(sample, key) for key in fields}, default=_json_default, allow_nan=False))


@dataclass(frozen=True)
class ExtractedFeature:
    representation: str
    values: np.ndarray
    samples: tuple[dict[str, Any], ...]
    split: str
    reader_name: str
    reader_version: str
    pooling: str
    checkpoint_reference: str

    def __post_init__(self) -> None:
        if not self.representation or self.split not in ("train", "validation", "test"):
            raise ValueError("representation and valid split are required")
        if self.pooling not in ("mean", "reader_provided"):
            raise ValueError("pooling must be mean or reader_provided")
        if not self.samples or self.values.ndim != 2 or self.values.shape[0] != len(self.samples):
            raise ValueError("feature rows and provenance must be non-empty and aligned")
        if self.values.dtype != np.float32:
            raise ValueError("feature dtype must be float32")
        if not np.isfinite(self.values).all():
            raise ValueError("features must be finite")


RepresentationReader = Callable[[torch.nn.Module, FeatureBatch], Mapping[str, Representation]]


def _pool(rep: Representation, pooling: str) -> tuple[torch.Tensor, str]:
    if pooling == "reader_provided":
        if rep.kind != "vector":
            raise ValueError("reader_provided requires a vector representation")
        return rep.tensor, "reader_provided"
    if pooling != "mean":
        raise ValueError("pooling must be mean or reader_provided")
    if rep.kind == "vector":
        return rep.tensor, "reader_provided"
    if rep.tensor.shape[1] == 0:
        raise ValueError("cannot mean-pool empty token dimension")
    return rep.tensor.mean(dim=1), "mean"


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: Iterable[FeatureBatch],
    *,
    reader: Any,
    context: SplitContext,
    representations: Sequence[str] | None = None,
    pooling: str = "mean",
    checkpoint_reference: str = "",
    device: str | torch.device = "cpu",
) -> dict[str, ExtractedFeature]:
    """Extract deterministic, provenance-aligned features in loader order."""
    if not getattr(reader, "name", None) or not getattr(reader, "version", None):
        raise ValueError("reader must expose name and version")
    if not isinstance(context, SplitContext):
        raise TypeError("extraction requires SplitContext")
    model = freeze_encoder(model)
    selected = tuple(representations) if representations is not None else None
    chunks: dict[str, list[np.ndarray]] = {}
    modes: dict[str, str] = {}
    sample_rows: list[dict[str, Any]] = []
    for batch in loader:
        if not isinstance(batch, FeatureBatch):
            raise TypeError("loader must yield FeatureBatch")
        for sample in batch.samples:
            if sample.subject_id not in context.subject_ids:
                raise SubjectLeakageError(f"subject {sample.subject_id!r} is outside {context.role} split")
        raw = reader.read(model, FeatureBatch(batch.signal.to(device), batch.samples))
        if not isinstance(raw, Mapping):
            raise TypeError("reader must return a mapping of representation names")
        names = selected or tuple(raw)
        for name in names:
            if name not in raw:
                raise ValueError(f"reader does not expose representation {name!r}")
            pooled, mode = _pool(raw[name], pooling)
            if name in modes and modes[name] != mode:
                raise ValueError(f"representation {name!r} changed pooling mode across batches")
            modes[name] = mode
            if pooled.shape[0] != len(batch.samples):
                raise ValueError(f"representation {name!r} is not aligned with provenance")
            chunks.setdefault(name, []).append(pooled.detach().float().cpu().numpy().astype(np.float32, copy=True))
        sample_rows.extend(sample_reference(sample) for sample in batch.samples)
    if not sample_rows:
        raise ValueError("cannot extract from an empty loader")
    result = {}
    for name, values in chunks.items():
        result[name] = ExtractedFeature(
            name, np.concatenate(values, axis=0), tuple(sample_rows), context.role,
            reader.name, reader.version, modes[name], checkpoint_reference,
        )
    return result
