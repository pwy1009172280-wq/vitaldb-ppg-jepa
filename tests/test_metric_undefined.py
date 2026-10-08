"""Required-metric-undefined fails closed (A11)."""

import numpy as np
import pytest

from src.downstream.core import PredictionBatch, evaluate_predictions
from src.downstream.core.metrics import MetricUndefinedError


def _single_class_prediction():
    return PredictionBatch(
        targets=np.array([0, 0]),
        predicted_labels=np.array([0, 0]),
        scores=np.array([[-1.0, -2.0], [-1.0, -2.0]]),
        probabilities=np.array([[0.9, 0.1], [0.9, 0.1]]),
        class_vocabulary=(0, 1), task="classification",
    )


def test_required_undefined_metric_fails_closed():
    p = _single_class_prediction()
    with pytest.raises(MetricUndefinedError, match="AUROC"):
        evaluate_predictions(p, ("accuracy", "AUROC"), required=("AUROC",))


def test_optional_undefined_metric_returns_none():
    p = _single_class_prediction()
    result = evaluate_predictions(p, ("accuracy", "AUROC"), required=("accuracy",))
    assert result["accuracy"] == 1.0
    assert result["AUROC"] is None


def test_all_metrics_required_by_default():
    p = _single_class_prediction()
    with pytest.raises(MetricUndefinedError):
        evaluate_predictions(p, ("AUROC",))
