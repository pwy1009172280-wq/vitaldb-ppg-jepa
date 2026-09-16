"""Configurable preprocessing for raw VitalDB PPG waveforms.

The baseline uses a fourth-order Chebyshev type-I, zero-phase band-pass filter,
non-overlapping windows, minimal finite/degenerate screening, and per-window
z-scoring.
Raw arrays are never modified in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal


@dataclass
class WindowQuality:
    quality_pass: bool
    rejection_reason: str | None
    flatline_fraction: float


@dataclass
class PPGWindow:
    signal: np.ndarray
    caseid: Any = None
    tid: Any = None
    window_index: int = 0
    start_sample: int = 0
    end_sample: int = 0  # exclusive
    sampling_rate: float = 500.0
    duration: float = 0.0
    quality_pass: bool | None = None
    rejection_reason: str | None = None
    flatline_fraction: float | None = None


def bandpass_filter_ppg(
    waveform: np.ndarray,
    sampling_rate: float = 500.0,
    low_cutoff: float = 0.5,
    high_cutoff: float = 12.0,
    order: int = 4,
    ripple_db: float = 0.5,
) -> np.ndarray:
    """Apply a fourth-order Chebyshev-I zero-phase band-pass filter.

    ``ripple_db`` (SciPy's ``rp`` parameter) is the pass-band ripple used by
    ``scipy.signal.cheby1``.
    ``sosfiltfilt`` avoids phase shifts.  A clear ``ValueError`` is raised when
    the input is too short for the required edge handling.
    """
    x = np.asarray(waveform, dtype=float)
    if x.ndim != 1:
        raise ValueError("waveform must be one-dimensional")
    if sampling_rate <= 0 or not (0 < low_cutoff < high_cutoff < sampling_rate / 2):
        raise ValueError("cutoffs must satisfy 0 < low < high < Nyquist")
    if order < 1 or ripple_db <= 0:
        raise ValueError("order must be positive and ripple_db must be positive")
    sos = signal.cheby1(
        order, ripple_db, [low_cutoff, high_cutoff], btype="bandpass",
        fs=sampling_rate, output="sos",
    )
    try:
        return signal.sosfiltfilt(sos, x)
    except ValueError as exc:
        raise ValueError("waveform is too short for zero-phase filtering") from exc


def split_into_windows(
    waveform: np.ndarray,
    sampling_rate: float = 500.0,
    window_seconds: float = 10.0,
    *,
    caseid: Any = None,
    tid: Any = None,
) -> list[PPGWindow]:
    """Split a waveform into non-overlapping complete windows.

    The incomplete trailing samples are discarded and never padded.  End
    indices use Python's exclusive convention.
    """
    x = np.asarray(waveform, dtype=float)
    if x.ndim != 1:
        raise ValueError("waveform must be one-dimensional")
    if sampling_rate <= 0 or window_seconds <= 0:
        raise ValueError("sampling_rate and window_seconds must be positive")
    samples_per_window = int(round(sampling_rate * window_seconds))
    if samples_per_window <= 0 or not np.isclose(samples_per_window, sampling_rate * window_seconds):
        raise ValueError("window_seconds must correspond to a whole number of samples")
    count = len(x) // samples_per_window
    return [
        PPGWindow(
            signal=x[i * samples_per_window : (i + 1) * samples_per_window].copy(),
            caseid=caseid, tid=tid, window_index=i,
            start_sample=i * samples_per_window,
            end_sample=(i + 1) * samples_per_window,
            sampling_rate=float(sampling_rate), duration=float(window_seconds),
        )
        for i in range(count)
    ]


def compute_flatline_fraction(waveform: np.ndarray) -> float:
    """Return the fraction of samples in exact constant runs of length >= 2.

    Equality is exact (no tolerance is invented).  Every sample belonging to a
    run of two or more identical adjacent values is counted as flatline.
    """
    x = np.asarray(waveform)
    if x.ndim != 1:
        raise ValueError("waveform must be one-dimensional")
    if len(x) == 0:
        return 0.0
    flat = np.zeros(len(x), dtype=bool)
    starts = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
    ends = np.r_[starts[1:], len(x)]
    for start, end in zip(starts, ends):
        if end - start >= 2:
            flat[start:end] = True
    return float(flat.mean())


def assess_window_quality(
    waveform: np.ndarray, epsilon: float = 1e-8
) -> WindowQuality:
    """Apply minimal v0 finite and near-constant screening."""
    x = np.asarray(waveform)
    if x.ndim != 1:
        raise ValueError("waveform must be one-dimensional")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not np.all(np.isfinite(x)):
        return WindowQuality(False, "non_finite", None)
    if float(np.std(x)) <= epsilon:
        return WindowQuality(False, "degenerate", None)
    return WindowQuality(True, None, None)


def zscore_normalize(waveform: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize one accepted window to mean zero and standard deviation one."""
    x = np.asarray(waveform, dtype=float)
    if x.ndim != 1:
        raise ValueError("waveform must be one-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("waveform must contain only finite values")
    mean = float(x.mean())
    std = float(x.std())
    if std <= epsilon:
        raise ValueError("cannot z-score a zero or near-zero variance waveform")
    return (x - mean) / std


def preprocess_ppg_record(
    waveform: np.ndarray,
    *,
    caseid: Any = None,
    tid: Any = None,
    sampling_rate: float = 500.0,
    window_seconds: float = 10.0,
    low_cutoff: float = 0.5,
    high_cutoff: float = 12.0,
    filter_order: int = 4,
    ripple_db: float = 0.5,
    epsilon: float = 1e-8,
) -> list[PPGWindow]:
    """Window first, then screen, filter, and normalize each raw PPG window.

    Filtering is independent per window, so a missing value in one window cannot
    propagate into any other window. All complete windows are returned so
    rejected windows retain their reason; accepted signals are float32.
    """
    windows = split_into_windows(
        np.asarray(waveform, dtype=float), sampling_rate, window_seconds,
        caseid=caseid, tid=tid,
    )
    for window in windows:
        quality = assess_window_quality(window.signal, epsilon)
        window.quality_pass = quality.quality_pass
        window.rejection_reason = quality.rejection_reason
        window.flatline_fraction = quality.flatline_fraction
        if not quality.quality_pass:
            continue
        try:
            filtered = bandpass_filter_ppg(
                window.signal, sampling_rate, low_cutoff, high_cutoff,
                filter_order, ripple_db,
            )
            if not np.all(np.isfinite(filtered)):
                window.quality_pass = False
                window.rejection_reason = "filtered_non_finite"
                continue
        except ValueError:
            window.quality_pass = False
            window.rejection_reason = "filtering_failed"
            continue
        try:
            normalized = zscore_normalize(filtered, epsilon)
        except ValueError:
            window.quality_pass = False
            window.rejection_reason = "normalization_failed"
            continue
        if not np.all(np.isfinite(normalized)):
            window.quality_pass = False
            window.rejection_reason = "normalization_nonfinite"
            continue
        window.signal = normalized.astype(np.float32)
    return windows
