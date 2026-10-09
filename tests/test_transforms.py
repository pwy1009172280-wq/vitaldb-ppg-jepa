"""Tests for the frozen preprocessing transforms (plan §6)."""

import numpy as np
import pytest

from src.preprocessing.transforms import (
    flatline_spans,
    preprocess_run,
    resample_poly_kaiser,
    valid_contiguous_runs,
    zscore_window,
)


def test_resample_500_to_125_length():
    x = np.sin(np.linspace(0, 10, 500))
    y = resample_poly_kaiser(x, up=1, down=4)
    assert y.dtype == np.float32
    assert len(y) == 125
    assert np.isfinite(y).all()


def test_zscore_unit_scale():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000).astype(np.float32) * 3 + 7
    z = zscore_window(x)
    assert z.dtype == np.float32
    assert abs(z.mean()) < 1e-5
    assert abs(z.std(ddof=0) - 1.0) < 1e-5


def test_zscore_rejects_degenerate():
    with pytest.raises(ValueError):
        zscore_window(np.ones(100))


def test_flatline_spans_detect_1s_constant():
    fs = 125.0
    x = np.random.default_rng(1).standard_normal(2000)
    x[100:225] = 5.0  # 125 samples = 1 s constant
    mask = flatline_spans(x, fs)
    assert mask[100:225].all()
    assert not mask[:100].any()


def test_valid_runs_split_at_gap_and_flatline():
    fs = 125.0
    x = np.arange(500, dtype=np.float32)
    x[100] = np.nan  # gap
    runs = valid_contiguous_runs(x, fs)
    assert runs == [(0, 100), (101, 500)]


def test_preprocess_run_125hz_identity_one_window():
    fs = 125.0
    # 2s trim + 16s window + 2s trim = 20s = 2500 samples
    rng = np.random.default_rng(2)
    x = rng.standard_normal(2500).astype(np.float32)
    result = preprocess_run(x, fs, native_rate=125.0)
    assert result.windows.shape == (1, 2000)
    assert abs(result.windows[0].mean()) < 1e-4
    assert abs(result.windows[0].std(ddof=0) - 1.0) < 1e-4


def test_preprocess_run_rejects_short():
    fs = 125.0
    x = np.zeros(100, dtype=np.float32)
    result = preprocess_run(x, fs, native_rate=125.0)
    assert result.windows.shape[0] == 0
