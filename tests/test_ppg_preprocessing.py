import numpy as np
import pytest
import src.data.ppg_preprocessing as ppg

from src.data.ppg_preprocessing import (
    assess_window_quality,
    bandpass_filter_ppg,
    compute_flatline_fraction,
    preprocess_ppg_record,
    split_into_windows,
    zscore_normalize,
)


def test_filter_preserves_length():
    x = np.sin(2 * np.pi * 2 * np.arange(5000) / 500)
    assert bandpass_filter_ppg(x).shape == x.shape


def test_window_sizes_and_trailing_samples():
    windows = split_into_windows(np.arange(10000 + 17), 500, 10)
    assert len(windows) == 2 and all(w.signal.size == 5000 for w in windows)
    assert (windows[0].start_sample, windows[0].end_sample) == (0, 5000)
    assert (windows[1].start_sample, windows[1].end_sample) == (5000, 10000)
    assert split_into_windows(np.arange(16000), 500, 8)[0].signal.size == 4000


def test_quality_nonfinite_and_flatline_threshold():
    x = np.arange(100, dtype=float)
    x[5] = np.nan
    assert not assess_window_quality(x).quality_pass
    assert assess_window_quality(np.r_[np.zeros(25), np.arange(1, 76, dtype=float)]).quality_pass
    assert not assess_window_quality(np.ones(100)).quality_pass
    assert not assess_window_quality(np.zeros(100)).quality_pass


def test_flatline_definition():
    assert compute_flatline_fraction(np.array([1, 1, 2, 3, 3, 3])) == pytest.approx(5 / 6)
    assert compute_flatline_fraction(np.arange(5)) == 0


def test_zscore_and_no_nan():
    y = zscore_normalize(np.array([1.0, 2.0, 3.0]))
    assert y.mean() == pytest.approx(0)
    assert y.std() == pytest.approx(1)
    assert np.isfinite(y).all()
    with pytest.raises(ValueError):
        zscore_normalize(np.ones(3))


def test_preprocess_provenance_and_normalization():
    t = np.arange(10000) / 500
    raw = np.sin(2 * np.pi * 2 * t)
    windows = preprocess_ppg_record(raw, caseid=12, tid=34)
    assert len(windows) == 2
    assert [(w.caseid, w.tid, w.window_index, w.start_sample, w.end_sample) for w in windows] == [
        (12, 34, 0, 0, 5000), (12, 34, 1, 5000, 10000)
    ]
    assert all(w.quality_pass and w.rejection_reason is None and w.signal.dtype == np.float32 for w in windows)
    assert all(w.signal.mean() == pytest.approx(0, abs=1e-6) and w.signal.std() == pytest.approx(1, abs=1e-6) for w in windows)
    assert all(w.duration == 10 and w.sampling_rate == 500 for w in windows)


def test_one_bad_window_does_not_invalidate_other_windows():
    t = np.arange(10000) / 500
    raw = np.sin(2 * np.pi * 2 * t)
    raw[:5000] = np.nan
    windows = preprocess_ppg_record(raw)
    assert windows[0].rejection_reason == "non_finite"
    assert windows[1].quality_pass
    assert np.isfinite(windows[1].signal).all()
    assert windows[1].signal.mean() == pytest.approx(0, abs=1e-6)
    assert windows[1].signal.std() == pytest.approx(1, abs=1e-6)


def test_inf_and_filter_length():
    t = np.arange(5000) / 500
    x = np.sin(2 * np.pi * 2 * t)
    x[0] = np.inf
    assert preprocess_ppg_record(x)[0].rejection_reason == "non_finite"
    y = preprocess_ppg_record(np.tile(x[1:], 2))
    assert all(w.signal.size == 5000 for w in y)


def test_normalization_nonfinite_is_rejected(monkeypatch):
    x = np.sin(2 * np.pi * 2 * np.arange(5000) / 500)
    monkeypatch.setattr(ppg, "zscore_normalize", lambda waveform, epsilon: np.full_like(waveform, np.nan))
    window = ppg.preprocess_ppg_record(x)[0]
    assert not window.quality_pass
    assert window.rejection_reason == "normalization_nonfinite"


def test_filter_edge_sanity_and_normal_preprocessing():
    t = np.arange(5000) / 500
    x = np.sin(2 * np.pi * 2 * t) + 0.1 * np.sin(2 * np.pi * 8 * t)
    filtered = bandpass_filter_ppg(x)
    assert filtered.shape == x.shape
    assert np.isfinite(filtered).all()
    assert np.isfinite(filtered[[0, -1]]).all()
    assert np.max(np.abs(filtered)) < 10
    window = preprocess_ppg_record(x)[0]
    assert window.quality_pass and window.signal.dtype == np.float32
