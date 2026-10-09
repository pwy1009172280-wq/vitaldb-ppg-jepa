"""Deterministic ridge solver (experiment plan §12): closed-form, float64 Cholesky."""

import numpy as np
import pytest

from src.downstream.core.ridge import RidgeRegression


def _design(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    w_true = np.arange(1, d + 1, dtype=np.float64)
    b_true = 3.7
    y = X @ w_true + b_true + 0.1 * rng.standard_normal(n)
    return X, y, w_true, b_true


def test_ridge_finite_predictions_and_solution():
    X, y, _, _ = _design()
    model = RidgeRegression(lambda_=0.001).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.isfinite(pred).all()
    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.intercept_)

    # normal-equation residual ‖A w − rhs‖ must be tiny (≤1e-8 relative)
    Xf = np.asarray(X, dtype=np.float64)
    mean = Xf.mean(axis=0)
    scale = Xf.std(axis=0, ddof=0)
    zero = scale <= 1e-8
    Z = (Xf - mean) / np.where(zero, 1.0, scale)
    Z[:, zero] = 0.0
    Zc = Z - Z.mean(axis=0)
    yc = y - y.mean()
    alpha = Xf.shape[0] * 0.001
    A = Zc.T @ Zc + alpha * np.eye(Xf.shape[1])
    rhs = Zc.T @ yc
    residual = np.linalg.norm(A @ model.coef_ - rhs)
    assert residual <= 1e-8 * max(1.0, np.linalg.norm(rhs))


def test_ridge_recovers_linear_target():
    X, y, w_true, b_true = _design()
    model = RidgeRegression(lambda_=1e-6).fit(X, y)
    pred = model.predict(X)
    # recovered predictions track the true linear target well
    assert np.mean(np.abs(pred - y)) < 0.2


def test_ridge_constant_dims_zeroed_and_recorded():
    X, y, _, _ = _design()
    X = np.concatenate([X, np.ones((X.shape[0], 2))], axis=1)  # two constant dims
    model = RidgeRegression().fit(X, y)
    assert set(model.zero_scale_dims) == {8, 9}
    assert np.allclose(model.coef_[8:], 0.0, atol=1e-12)
    assert model.predict(X).shape == (X.shape[0],)


def test_ridge_rejects_invalid_input():
    with pytest.raises(ValueError):
        RidgeRegression(lambda_=0)
    with pytest.raises(ValueError):
        RidgeRegression().fit(np.ones((5, 3)), np.ones(4))
    with pytest.raises(ValueError):
        RidgeRegression().fit(np.full((5, 3), np.nan), np.ones(5))
