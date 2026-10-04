"""Dataset-agnostic sample contracts for biosignal experiments."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class UnifiedSample:
    """A single time-window with provenance shared by every dataset adapter.

    ``subject_id`` is intentionally mandatory: split code must be able to
    group samples by subject before any model-facing batching takes place.
    ``signal`` uses the canonical ``(channels, time)`` layout.
    """

    signal: np.ndarray
    subject_id: str
    recording_id: str
    dataset: str
    modality: str
    sampling_rate_hz: float
    start_time_s: float
    end_time_s: float
    channel_names: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    valid_mask: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    window_id: str | None = None
    window_start_sample: int | None = None
    window_end_sample: int | None = None
    labels: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        signal = np.asarray(self.signal)
        if signal.ndim != 2:
            raise ValueError("signal must have shape (channels, time)")
        if not self.subject_id:
            raise ValueError("subject_id must be non-empty")
        if not self.recording_id:
            raise ValueError("recording_id must be non-empty")
        if not self.dataset or not self.modality:
            raise ValueError("dataset and modality must be non-empty")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be > 0")
        if self.end_time_s <= self.start_time_s:
            raise ValueError("end_time_s must be greater than start_time_s")
        if self.channel_names and len(self.channel_names) != signal.shape[0]:
            raise ValueError("channel_names must match signal channels")
        if self.units and len(self.units) != signal.shape[0]:
            raise ValueError("units must match signal channels")
        if self.valid_mask is not None:
            valid_mask = np.asarray(self.valid_mask)
            if valid_mask.shape not in (signal.shape, signal.shape[1:]):
                raise ValueError("valid_mask must have shape (time) or (channels, time)")
            if valid_mask.dtype != np.bool_:
                raise ValueError("valid_mask must have boolean dtype")
        if self.window_start_sample is not None and self.window_start_sample < 0:
            raise ValueError("window_start_sample must be non-negative")
        if self.window_end_sample is not None and self.window_end_sample <= 0:
            raise ValueError("window_end_sample must be positive")
        if (
            self.window_start_sample is not None
            and self.window_end_sample is not None
            and self.window_end_sample <= self.window_start_sample
        ):
            raise ValueError("window_end_sample must exceed window_start_sample")
        if not np.isfinite(signal).all():
            raise ValueError("signal must contain only finite values")

    @property
    def num_channels(self) -> int:
        return int(self.signal.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.signal.shape[1])
