"""Deterministic ridge linear regression solver (frozen experiment plan §12 recipe).

Objective: (1/n) Σ_i (y_i − b − z_iᵀw)² + λ·‖w‖²₂

- intercept ``b`` is NOT regularized;
- per-dimension population standard deviation standardization, dims with
  std ≤ 1e-8 are fixed to zero AND recorded (never dropped: all 128 dims stay);
- target is never standardized (stays in bpm);
- CPU float64 deterministic Cholesky solve of the normal equations;
- fixed λ = 0.001, equivalently alpha = n_train × 0.001 for a library whose
  objective is the unnormalized SSE.
"""

from __future__ import annotations

import numpy as np


class RidgeRegression:
    """Closed-form ridge linear regression with intercept (128 → 1)."""

    def __init__(self, lambda_: float = 0.001) -> None:
        if lambda_ <= 0:
            raise ValueError("ridge lambda must be positive")
        self.lambda_ = float(lambda_)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.zero_scale_dims: tuple[int, ...] = ()
        self.alpha: float | None = None

    def fit(self, features, targets) -> "RidgeRegression":
        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64).ravel()
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("ridge fit requires [samples, features]")
        if y.shape[0] != X.shape[0]:
            raise ValueError("feature and target sample counts differ")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("ridge fit requires finite features and targets")

        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0, ddof=0)  # population std
        zero = self.scale <= 1e-8
        self.zero_scale_dims = tuple(int(i) for i in np.flatnonzero(zero))
        scale_safe = np.where(zero, 1.0, self.scale)

        Z = (X - self.mean) / scale_safe
        Z[:, zero] = 0.0

        z_mean = Z.mean(axis=0)
        Zc = Z - z_mean
        yc = y - y.mean()

        n = X.shape[0]
        self.alpha = n * self.lambda_
        dim = X.shape[1]
        A = Zc.T @ Zc + self.alpha * np.eye(dim)
        rhs = Zc.T @ yc
        L = np.linalg.cholesky(A)
        w = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        self.coef_ = w
        self.intercept_ = float(y.mean() - w @ z_mean)
        return self

    def predict(self, features) -> np.ndarray:
        if self.coef_ is None or self.mean is None or self.scale is None:
            raise RuntimeError("RidgeRegression must be fit before predict")
        X = np.asarray(features, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.mean):
            raise ValueError("feature dimensions do not match fit")
        zero = self.scale <= 1e-8
        scale_safe = np.where(zero, 1.0, self.scale)
        Z = (X - self.mean) / scale_safe
        Z[:, zero] = 0.0
        return Z @ self.coef_ + self.intercept_
