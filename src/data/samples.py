"""Dataset-agnostic sample contracts for biosignal experiments."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


SUBJECT_IDENTITY_RESOLVED = "RESOLVED"
SUBJECT_IDENTITY_UNRESOLVED = "SUBJECT_IDENTITY_UNRESOLVED"
SUBJECT_IDENTITY_STATUSES = (SUBJECT_IDENTITY_RESOLVED, SUBJECT_IDENTITY_UNRESOLVED)


@dataclass(frozen=True)
class UnifiedSample:
    """A single time-window with provenance shared by every dataset adapter.

    ``subject_id`` is a stable namespaced canonical patient ID only when
    ``subject_identity_status == RESOLVED``; it is ``None`` otherwise. Split
    and leakage code must reject unresolved subjects rather than guess.

    ``signal`` uses the canonical ``(channels, time)`` layout.
    """

    signal: np.ndarray
    subject_id: str | None
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
    subject_identity_status: str = SUBJECT_IDENTITY_UNRESOLVED
    subject_identity_namespace: str | None = None
    subject_identity_kind: str | None = None
    subject_source_identity: str | None = None
    subject_identity_mapping_ref: str | None = None

    def __post_init__(self) -> None:
        signal = np.asarray(self.signal)
        if signal.ndim != 2:
            raise ValueError("signal must have shape (channels, time)")
        if not self.recording_id:
            raise ValueError("recording_id must be non-empty")
        if not self.dataset or not self.modality:
            raise ValueError("dataset and modality must be non-empty")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be > 0")
        if self.end_time_s <= self.start_time_s:
            raise ValueError("end_time_s must be greater than start_time_s")
        if self.subject_identity_status not in SUBJECT_IDENTITY_STATUSES:
            raise ValueError(
                f"subject_identity_status must be one of {SUBJECT_IDENTITY_STATUSES}"
            )
        if self.subject_identity_status == SUBJECT_IDENTITY_RESOLVED:
            if not isinstance(self.subject_id, str) or not self.subject_id:
                raise ValueError("RESOLVED sample requires a non-empty subject_id")
        else:
            if self.subject_id is not None:
                raise ValueError(
                    "SUBJECT_IDENTITY_UNRESOLVED sample must have subject_id=None; "
                    "carry the raw source identifier in subject_source_identity"
                )
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
        self._validate_finiteness(signal)

    def _validate_finiteness(self, signal: np.ndarray) -> None:
        finite = np.isfinite(signal)
        if self.valid_mask is None:
            invalid = ~finite
        else:
            mask = np.asarray(self.valid_mask)
            if mask.shape == signal.shape[1:]:
                mask_full = np.broadcast_to(mask, signal.shape)
            else:
                mask_full = mask
            # non-finite values are permitted only at explicitly invalid positions
            invalid = ~finite & mask_full
        if invalid.any():
            raise ValueError("signal contains non-finite values at valid positions")

    @property
    def num_channels(self) -> int:
        return int(self.signal.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.signal.shape[1])
