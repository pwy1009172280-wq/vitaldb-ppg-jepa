"""Metrics with explicit prediction-input requirements.

Required metrics that are undefined for the given predictions fail closed with
:class:`MetricUndefinedError` (data degeneration, e.g. a single class, is never
silently reported as success). Optional diagnostics may return ``None`` and are
excluded from the required set by the caller.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from scipy.stats import pearsonr, spearmanr

from .probe import PredictionBatch


class MetricUndefinedError(ValueError):
    """A required metric cannot be computed for the given predictions."""

    def __init__(self, metric: str, reason: str) -> None:
        self.metric = metric
        self.reason = reason
        super().__init__(f"required metric {metric!r} is undefined: {reason}")


def _classification_metric(prediction: PredictionBatch, metric: str, target: np.ndarray):
    predicted = np.asarray(prediction.predicted_labels)
    if metric == "accuracy":
        return float(accuracy_score(target, predicted))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(target, predicted))
    if metric == "macro_f1":
        return float(f1_score(target, predicted, average="macro", zero_division=0))
    if prediction.probabilities is None or len(np.unique(target)) < 2:
        return None
    try:
        if len(prediction.class_vocabulary) == 2:
            scores = prediction.probabilities[:, 1]
            return float(roc_auc_score(target, scores) if metric == "AUROC" else average_precision_score(target, scores))
        if metric == "AUROC":
            return float(roc_auc_score(target, prediction.probabilities, multi_class="ovr", average="macro"))
        one_vs_rest = label_binarize(target, classes=np.arange(len(prediction.class_vocabulary)))
        return float(average_precision_score(one_vs_rest, prediction.probabilities, average="macro"))
    except ValueError:
        return None


def _regression_metric(prediction: PredictionBatch, metric: str, target: np.ndarray):
    scores = np.asarray(prediction.scores, dtype=np.float64)
    if metric == "MAE":
        return float(mean_absolute_error(target, scores))
    if metric == "RMSE":
        return float(np.sqrt(mean_squared_error(target, scores)))
    if len(target) < 2 or np.ptp(target) == 0 or np.ptp(scores) == 0:
        return None
    if metric == "Pearson":
        return float(pearsonr(target, scores).statistic)
    if metric == "Spearman":
        return float(spearmanr(target, scores).statistic)
    return None


def evaluate_predictions(prediction: PredictionBatch, metrics: tuple[str, ...], *, required: tuple[str, ...] | None = None) -> dict[str, float | None]:
    """Compute metrics; fail closed when a required metric is undefined.

    ``required`` defaults to ``metrics`` (every requested metric is required).
    Pass an explicit subset to allow optional diagnostics to be ``None``.
    """
    if prediction.task == "classification":
        supported = {"accuracy", "balanced_accuracy", "macro_f1", "AUROC", "AUPRC"}
    elif prediction.task == "regression":
        supported = {"MAE", "RMSE", "Pearson", "Spearman"}
    else:
        raise ValueError("prediction task must be classification or regression")
    if any(metric not in supported for metric in metrics):
        raise ValueError("unsupported metric for prediction task")
    required_set = set(required) if required is not None else set(metrics)
    target = np.asarray(prediction.targets)
    result: dict[str, float | None] = {}
    for metric in metrics:
        if prediction.task == "classification":
            value = _classification_metric(prediction, metric, target)
        else:
            value = _regression_metric(prediction, metric, target)
        if value is None and metric in required_set:
            raise MetricUndefinedError(metric, "undefined for the given predictions")
        result[metric] = value
    return result
