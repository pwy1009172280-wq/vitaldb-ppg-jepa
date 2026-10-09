"""Tests for VitalDB PLETH parsing + Solar8000/HR target (plan §9.1)."""

import numpy as np
import pytest

from src.experiments.vitaldb_preprocess import compute_hr_target, parse_pleth_csv


def test_parse_pleth_csv_500hz_nan():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("Time,SNUADC/PLETH\n0,\n0.002,1.0\n0.004,2.0\n0.006,\n")
        path = f.name
    sig, fs = parse_pleth_csv(path)
    assert fs == pytest.approx(500.0, rel=1e-3)
    assert sig.shape == (4,)
    assert np.isnan(sig[0])
    assert sig[1] == 1.0
    assert np.isnan(sig[3])


def test_parse_pleth_sparse_implicit_time():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("Time,SNUADC/PLETH\n0,\n0.002,\n,1.0\n,2.0\n,3.0\n")
        path = f.name
    sig, fs = parse_pleth_csv(path)
    assert fs == pytest.approx(500.0, rel=1e-3)
    # 5 data rows: NaN, NaN, 1.0, 2.0, 3.0
    assert sig.shape == (5,)
    assert np.isnan(sig[0]) and np.isnan(sig[1])
    assert list(sig[2:]) == [1.0, 2.0, 3.0]


def test_hr_full_coverage_weighted_median():
    t = np.array([0.0, 4.0, 8.0, 12.0])
    v = np.array([60.0, 70.0, 80.0, 90.0])
    r = compute_hr_target(t, v, 0.0)
    assert r is not None
    target, coverage, n, span = r
    assert target == 70.0
    assert coverage == pytest.approx(1.0)
    assert n == 4


def test_hr_requires_two_valid():
    t = np.array([0.0, 2.0])
    v = np.array([70.0, 999.0])  # only 1 valid
    assert compute_hr_target(t, v, 0.0) is None


def test_hr_rejects_short_span():
    t = np.array([0.0, 2.0, 4.0])
    v = np.array([60.0, 70.0, 80.0])
    # first-last valid span = 4s < 8s age_limit
    assert compute_hr_target(t, v, 0.0) is None


def test_hr_rejects_insufficient_coverage():
    t = np.array([0.0, 20.0])
    v = np.array([60.0, 70.0])
    # first holds [0,8], then gap -> coverage 8/16 = 50%
    assert compute_hr_target(t, v, 0.0) is None


def test_hr_rejects_out_of_range():
    t = np.array([0.0, 4.0, 8.0, 12.0])
    v = np.array([60.0, 10.0, 80.0, 90.0])  # 10 bpm out of [30,220]
    r = compute_hr_target(t, v, 0.0)
    # 3 valid updates remain but coverage may still be computed
    assert r is None or r[1] < 0.9
