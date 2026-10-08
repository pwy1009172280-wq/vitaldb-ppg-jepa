"""Frozen extraction, explicit sample provenance, and pickle-free caches."""

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
from .config import validate_layers


@dataclass(frozen=True)
class FeatureBatch:
    signal: torch.Tensor
    samples: tuple[UnifiedSample, ...]

    def __post_init__(self):
        if not isinstance(self.signal, torch.Tensor) or not self.samples or any(not isinstance(sample, UnifiedSample) for sample in self.samples):
            raise TypeError("FeatureBatch requires a Tensor and UnifiedSample provenance")
        if self.signal.ndim != 3 or self.signal.shape[0] != len(self.samples):
            raise ValueError("FeatureBatch requires [batch, channels, time] and aligned samples")


def collate_feature_samples(samples: Sequence[UnifiedSample]) -> FeatureBatch:
    """Collate fixed-sized, already preprocessed windows; never pad/augment."""
    if not samples or any(not isinstance(sample, UnifiedSample) for sample in samples):
        raise TypeError("collation requires non-empty UnifiedSample batches")
    shapes = {np.asarray(sample.signal).shape for sample in samples}
    if len(shapes) != 1:
        raise ValueError("signal shapes differ; preprocessing must resolve this before extraction")
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


def _sample_reference(sample: UnifiedSample) -> dict[str, Any]:
    # Deliberately omit the waveform and validity-mask arrays from a feature cache.
    fields = (
        "subject_id", "recording_id", "dataset", "modality", "sampling_rate_hz",
        "start_time_s", "end_time_s", "channel_names", "units", "window_id",
        "window_start_sample", "window_end_sample", "provenance", "metadata", "labels",
    )
    return json.loads(json.dumps({key: getattr(sample, key) for key in fields}, default=_json_default, allow_nan=False))


def _sample_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in (
        "dataset", "subject_id", "recording_id", "window_id", "window_start_sample",
        "window_end_sample", "start_time_s", "end_time_s",
    ))


@dataclass(frozen=True)
class ExtractedFeatures:
    features: dict[str, np.ndarray]
    samples: tuple[dict[str, Any], ...]
    split: str
    checkpoint_reference: str
    pooling: dict[str, str]

    def __post_init__(self):
        validate_layers(tuple(self.features))
        if set(self.pooling) != set(self.features) or any(mode not in ("mean_tokens", "reader_prepooled") for mode in self.pooling.values()):
            raise ValueError("every feature layer must declare its readout")
        if self.split not in ("train", "validation", "test") or not self.checkpoint_reference or not self.samples:
            raise ValueError("non-empty samples, checkpoint reference, and valid split are required")
        for row in self.samples:
            if any(not row.get(key) for key in ("subject_id", "recording_id", "dataset")):
                raise ValueError("sample/subject provenance is required")
        keys = [_sample_key(row) for row in self.samples]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate sample provenance")
        for name, array in self.features.items():
            if not isinstance(array, np.ndarray) or array.ndim != 2 or array.shape[0] != len(self.samples) or array.shape[1] == 0:
                raise ValueError(f"unaligned or invalid feature matrix: {name}")
            if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                raise ValueError("features must be finite floating-point matrices")

    @property
    def subject_ids(self) -> frozenset[str]:
        return frozenset(row["subject_id"] for row in self.samples)


EncoderFactory = Callable[[Mapping[str, Any]], torch.nn.Module]
RepresentationReader = Callable[[torch.nn.Module, FeatureBatch], Mapping[str, torch.Tensor]]


def load_frozen_encoder(checkpoint: str | Path, encoder_factory: EncoderFactory, *, device="cpu") -> torch.nn.Module:
    """Load a trusted existing checkpoint; construction remains caller-owned.

    Both existing training checkpoint formats expose their state under 'model'.
    The factory receives the payload to use its resolved architecture config.
    Checkpoints are trusted inputs because legacy payloads need weights_only=False.
    """
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError("checkpoint must contain a model state dictionary")
    encoder = encoder_factory(payload)
    if not isinstance(encoder, torch.nn.Module):
        raise TypeError("encoder_factory must return torch.nn.Module")
    encoder.load_state_dict(payload["model"], strict=True)
    encoder.to(device).eval().requires_grad_(False)
    return encoder


def read_encoder_depths(encoder: torch.nn.Module, batch: FeatureBatch) -> Mapping[str, torch.Tensor]:
    """Read the existing encode_full contract, without importing any SSL model.

    layer_N is block N's output; final_layer includes the encoder's final norm.
    A different encoder/batch convention can supply a RepresentationReader.
    """
    if any(sample.valid_mask is not None and not np.asarray(sample.valid_mask).all() for sample in batch.samples):
        raise ValueError("default encode_full reader does not support invalid/padded samples")
    representation = encoder.encode_full(batch.signal)
    result = {f"layer_{index + 1}": tensor for index, tensor in enumerate(representation.hidden_states)}
    result["final_layer"] = representation.final_tokens
    return result


@torch.no_grad()
def extract_layer_features(
    encoder: torch.nn.Module,
    loader: Iterable[FeatureBatch],
    *,
    layers: tuple[str, ...],
    context: SplitContext,
    checkpoint_reference: str,
    device="cpu",
    representation_reader: RepresentationReader = read_encoder_depths,
) -> ExtractedFeatures:
    validate_layers(layers)
    if not isinstance(context, SplitContext) or context.role not in ("train", "validation", "test"):
        raise ValueError("extraction requires a valid SplitContext")
    if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("encoder must already be in eval mode with all parameters frozen")
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in layers}
    pooling: dict[str, str] = {}
    samples = []
    for batch in loader:
        if not isinstance(batch, FeatureBatch):
            raise TypeError("loader must yield FeatureBatch with UnifiedSample provenance")
        for sample in batch.samples:
            if sample.subject_id not in context.subject_ids:
                raise SubjectLeakageError(f"subject {sample.subject_id!r} is outside {context.role} split")
        tensors = representation_reader(encoder, FeatureBatch(batch.signal.to(device), batch.samples))
        for name in layers:
            if name not in tensors:
                raise ValueError(f"encoder does not expose requested layer {name!r}")
            tensor = tensors[name]
            if tensor.ndim == 3 and tensor.shape[1] > 0:
                mode = "mean_tokens"
                tensor = tensor.mean(dim=1)
            elif tensor.ndim == 2:
                mode = "reader_prepooled"
            else:
                raise ValueError("representations must be [batch, tokens, dim] or [batch, dim]")
            if name in pooling and pooling[name] != mode:
                raise ValueError(f"inconsistent readout across batches for {name}")
            pooling[name] = mode
            if tensor.shape[0] != len(batch.samples):
                raise ValueError("representation batch size does not match sample provenance")
            chunks[name].append(tensor.detach().float().cpu().numpy().copy())
        samples.extend(_sample_reference(sample) for sample in batch.samples)
    if not samples:
        raise ValueError(f"{context.role} loader is empty")
    return ExtractedFeatures(
        {name: np.concatenate(values) for name, values in chunks.items()}, tuple(samples),
        context.role, str(checkpoint_reference), pooling,
    )


def save_extracted_features(bundle: ExtractedFeatures, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1, "layers": list(bundle.features), "samples": bundle.samples,
        "split": bundle.split, "checkpoint_reference": bundle.checkpoint_reference,
        "pooling": bundle.pooling,
    }
    with path.open("wb") as handle:
        np.savez_compressed(handle, metadata_json=np.asarray(json.dumps(metadata, allow_nan=False, default=_json_default)), **bundle.features)
    return path


def load_extracted_features(path: str | Path) -> ExtractedFeatures:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata["format_version"] != 1:
            raise ValueError("unsupported extracted-feature format")
        return ExtractedFeatures(
            {name: archive[name].copy() for name in metadata["layers"]},
            tuple(metadata["samples"]), metadata["split"], metadata["checkpoint_reference"], metadata["pooling"],
        )
