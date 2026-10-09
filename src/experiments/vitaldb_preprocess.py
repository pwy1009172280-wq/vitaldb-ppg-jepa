"""VitalDB preprocessing + Solar8000/HR target (frozen plan §6.1 + §9.1)."""

from __future__ import annotations

import numpy as np

HR_RANGE = (30.0, 220.0)
AGE_LIMIT = 8.0
COVERAGE_MIN = 0.90
WINDOW_SECONDS = 16.0


def parse_pleth_csv(path: str) -> tuple[np.ndarray, float]:
    """Parse a VitalDB SNUADC/PLETH track CSV into (signal, fs).

    The waveform CSV is a continuous grid at 500 Hz with IMPLICIT time
    (row_index × interval) and sparse explicit time cells (start/end only).
    An empty value cell is NaN (device not recording / invalid).
    """
    values: list[float] = []
    first_times: list[float] = []
    with open(path) as f:
        f.readline()  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            tcell = parts[0].strip() if parts else ""
            vcell = parts[1].strip() if len(parts) > 1 else ""
            if tcell != "" and len(first_times) < 2:
                first_times.append(float(tcell))
            values.append(float(vcell) if vcell != "" else np.nan)
    if len(first_times) >= 2 and first_times[1] > first_times[0]:
        fs = 1.0 / (first_times[1] - first_times[0])
    else:
        fs = 500.0
    return np.asarray(values, dtype=np.float32), float(fs)


def compute_hr_target(
    hr_times: np.ndarray,
    hr_values: np.ndarray,
    window_start: float,
    *,
    window_seconds: float = WINDOW_SECONDS,
    age_limit: float = AGE_LIMIT,
    coverage_min: float = COVERAGE_MIN,
) -> tuple[float, float, int, float] | None:
    """Compute the duration-weighted-median HR target for a 16 s window.

    Returns (target_bpm, coverage, n_valid_updates, first_last_span) or None if
    target-ineligible. Implements the frozen zero-order-hold label rule.
    """
    t = np.asarray(hr_times, dtype=np.float64)
    v = np.asarray(hr_values, dtype=np.float64)
    window_end = window_start + window_seconds

    order = np.argsort(t)
    t = t[order]
    v = v[order]

    lo, hi = HR_RANGE
    valid = np.isfinite(v) & (v >= lo) & (v <= hi)
    if valid.sum() < 2:
        return None

    vt = t[valid]
    if vt[-1] - vt[0] < age_limit:
        return None

    # Build zero-order-hold coverage within the window. Each valid update at t[j]
    # holds from t[j] until min(next update (valid OR invalid), t[j]+age_limit).
    # Invalid updates terminate the previous hold and contribute no hold.
    intervals: list[tuple[float, float, float]] = []  # (start, end, value)
    valid_idx = np.flatnonzero(valid)
    for j in valid_idx:
        start = float(t[j])
        end = start + age_limit
        if j + 1 < t.size:
            end = min(end, float(t[j + 1]))
        if end <= start:
            continue
        s = max(start, window_start)
        e = min(end, window_end)
        if e > s:
            intervals.append((s, e, float(v[j])))

    if not intervals:
        return None

    covered = 0.0
    merged: list[tuple[float, float, float]] = []
    for s, e, val in sorted(intervals):
        covered += e - s
        merged.append((s, e, val))

    coverage = covered / window_seconds
    if coverage < coverage_min:
        return None

    # duration-weighted median: sort by value, cumulative duration, smallest value
    # reaching 50% cumulative weight.
    merged.sort(key=lambda x: x[2])
    total = sum(e - s for s, e, _ in merged)
    half = total / 2.0
    cum = 0.0
    for s, e, val in merged:
        cum += e - s
        if cum >= half:
            target = float(val)
            break
    else:
        target = float(merged[-1][2])

    return target, coverage, int(valid.sum()), float(vt[-1] - vt[0])
