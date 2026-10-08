"""Metrics with explicit prediction-input requirements."""

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


def evaluate_predictions(prediction: PredictionBatch, metrics: tuple[str, ...]) -> dict[str, float | None]:
    if prediction.task == "classification":
        supported = {"accuracy", "balanced_accuracy", "macro_f1", "AUROC", "AUPRC"}
    elif prediction.task == "regression":
        supported = {"MAE", "RMSE", "Pearson", "Spearman"}
    else:
        raise ValueError("prediction task must be classification or regression")
    if any(metric not in supported for metric in metrics):
        raise ValueError("unsupported metric for prediction task")
    target = np.asarray(prediction.targets)
    result: dict[str, float | None] = {}
    if prediction.task == "classification":
        if prediction.predicted_labels is None:
            raise ValueError("classification metrics require predicted_labels")
        predicted = np.asarray(prediction.predicted_labels)
        for metric in metrics:
            if metric == "accuracy":
                result[metric] = float(accuracy_score(target, predicted))
            elif metric == "balanced_accuracy":
                result[metric] = float(balanced_accuracy_score(target, predicted))
            elif metric == "macro_f1":
                result[metric] = float(f1_score(target, predicted, average="macro", zero_division=0))
            else:
                if prediction.probabilities is None or len(np.unique(target)) < 2:
                    result[metric] = None
                    continue
                try:
                    if len(prediction.class_vocabulary) == 2:
                        scores = prediction.probabilities[:, 1]
                        result[metric] = float(roc_auc_score(target, scores) if metric == "AUROC" else average_precision_score(target, scores))
                    else:
                        if metric == "AUROC":
                            result[metric] = float(roc_auc_score(target, prediction.probabilities, multi_class="ovr", average="macro"))
                        else:
                            one_vs_rest = label_binarize(target, classes=np.arange(len(prediction.class_vocabulary)))
                            result[metric] = float(average_precision_score(one_vs_rest, prediction.probabilities, average="macro"))
                except ValueError:
                    result[metric] = None
        return result

    scores = np.asarray(prediction.scores, dtype=np.float64) if prediction.scores is not None else None
    if scores is None:
        raise ValueError("regression metrics require scores")
    for metric in metrics:
        if metric == "MAE":
            result[metric] = float(mean_absolute_error(target, scores))
        elif metric == "RMSE":
            result[metric] = float(np.sqrt(mean_squared_error(target, scores)))
        elif metric == "Pearson":
            result[metric] = None if len(target) < 2 or np.ptp(target) == 0 or np.ptp(scores) == 0 else float(pearsonr(target, scores).statistic)
        else:
            result[metric] = None if len(target) < 2 or np.ptp(target) == 0 or np.ptp(scores) == 0 else float(spearmanr(target, scores).statistic)
    return result
