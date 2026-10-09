"""Preprocessing transforms for the frozen experiment recipe (plan §6).

Native decode → continuity/gap split → per-run anti-alias resample (Kaiser 5.0)
→ trim 2 s each end → fixed 16 s non-overlap window grid → window QC → per-window
z-score → float32. No bandpass, no interpolation, no zero-padding, no mirror fill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

WINDOW_SECONDS = 16.0
TRIM_SECONDS = 2.0
TARGET_RATE = 125.0
FLATLINE_SECONDS = 1.0


def resample_poly_kaiser(x: np.ndarray, *, up: int, down: int) -> np.ndarray:
    """Anti-alias polyphase resample with a Kaiser (β=5.0) window."""
    if up <= 0 or down <= 0:
        raise ValueError("up/down must be positive")
    values = np.asarray(x, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("resample requires finite input")
    return signal.resample_poly(values, up, down, window=("kaiser", 5.0)).astype(np.float32)


def zscore_window(x: np.ndarray) -> np.ndarray:
    """Per-window (x − mean) / population_std, float64 compute, float32 output."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("zscore requires a non-empty 1-D window")
    std = values.std(ddof=0)
    if std <= 1e-8:
        raise ValueError("window std <= 1e-8 (degenerate window rejected)")
    return ((values - values.mean()) / std).astype(np.float32)


def flatline_spans(x: np.ndarray, fs: float) -> np.ndarray:
    """Return a boolean mask of samples inside a ≥1 s constant (flatline) span."""
    values = np.asarray(x)
    n = int(round(fs * FLATLINE_SECONDS))
    if n < 2:
        return np.zeros(values.size, dtype=bool)
    # A run of equal values of length >= n marks a flatline.
    mask = np.zeros(values.size, dtype=bool)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[j + 1] == values[j]:
            j += 1
        if j - i + 1 >= n:
            mask[i : j + 1] = True
        i = j + 1
    return mask


def valid_contiguous_runs(x: np.ndarray, fs: float) -> list[tuple[int, int]]:
    """Return [start, end) spans of finite samples that avoid flatline spans.

    The plan forbids interpolation/padding across gaps; a ≥1 s constant span is
    treated as a break (invalid interval).
    """
    values = np.asarray(x)
    finite = np.isfinite(values)
    flat = flatline_spans(values, fs)
    valid = finite & ~flat
    runs: list[tuple[int, int]] = []
    i = 0
    while i < valid.size:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < valid.size and valid[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


@dataclass
class WindowResult:
    windows: np.ndarray  # [n_windows, window_samples]
    start_samples: list[int]
    n_rejected: int
    n_flatline: int
    n_gap: int


def preprocess_run(signal: np.ndarray, fs: float, *, native_rate: float) -> WindowResult:
    """Process one native continuous run into 16 s z-scored windows at 125 Hz.

    ``signal`` is the raw run (already continuous at native_rate). If native_rate
    is not 125, anti-alias resample to 125 Hz first. Then trim 2 s each end,
    grid 16 s non-overlap windows, z-score each, and count rejections.
    """
    values = np.asarray(signal, dtype=np.float32).ravel()
    if not np.isfinite(values).all():
        raise ValueError("preprocess_run requires a finite continuous run")

    if native_rate == TARGET_RATE:
        resampled = values
        out_fs = TARGET_RATE
    elif native_rate % TARGET_RATE == 0:
        down = int(round(native_rate / TARGET_RATE))
        resampled = resample_poly_kaiser(values, up=1, down=down)
        out_fs = TARGET_RATE
    else:
        raise ValueError(f"unsupported native rate {native_rate} for target 125 Hz")

    trim = int(round(TRIM_SECONDS * out_fs))
    if resampled.size < 2 * trim + int(round(WINDOW_SECONDS * out_fs)):
        return WindowResult(np.zeros((0, int(round(WINDOW_SECONDS * out_fs))), dtype=np.float32), [], 0, 0, 0)

    core = resampled[trim : resampled.size - trim]
    win_len = int(round(WINDOW_SECONDS * out_fs))  # 2000
    n_windows = core.size // win_len
    windows: list[np.ndarray] = []
    starts: list[int] = []
    rejected = 0
    for w in range(n_windows):
        seg = core[w * win_len : (w + 1) * win_len]
        if not np.isfinite(seg).all():
            rejected += 1
            continue
        if float(np.asarray(seg, dtype=np.float64).std(ddof=0)) <= 1e-8:
            rejected += 1
            continue
        windows.append(zscore_window(seg))
        starts.append(trim + w * win_len)
    arr = np.stack(windows).astype(np.float32) if windows else np.zeros((0, win_len), dtype=np.float32)
    return WindowResult(arr, starts, rejected, 0, 0)
