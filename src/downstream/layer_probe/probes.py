"""Linear prediction heads over detached features, using the existing Trainer."""

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...training.contracts import LossOutput
from .config import SUPPORTED_METRICS


class LinearProbe(nn.Module):
    """No encoder reference: only the head's weight and bias are trainable.

    Training-only normalization and class vocabulary are checkpointed with the
    head, so a saved probe is not detached from its feature preprocessing.
    """

    def __init__(self, train_features: np.ndarray, train_targets: np.ndarray, *, task: str, standardize: bool = True):
        super().__init__()
        x = np.asarray(train_features, dtype=np.float32)
        y = np.asarray(train_targets)
        if x.ndim != 2 or not len(x) or x.shape[1] == 0 or not np.isfinite(x).all() or y.shape != (len(x),):
            raise ValueError("training features and targets must be finite and aligned")
        if task not in SUPPORTED_METRICS:
            raise ValueError("task must be classification or regression")
        self.task = task
        self.classes: tuple = ()
        if task == "classification":
            if y.dtype.kind not in "biufUS" or (y.dtype.kind in "biuf" and not np.isfinite(y).all()):
                raise ValueError("classification targets must be finite scalar numbers or strings")
            self.classes = tuple(np.unique(y).tolist())
            if len(self.classes) < 2:
                raise ValueError("classification training requires at least two classes")
            outputs = len(self.classes)
        else:
            if not np.isfinite(y.astype(np.float64)).all():
                raise ValueError("regression targets must be finite")
            outputs = 1
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32) if standardize else np.zeros(x.shape[1], dtype=np.float32)
        scale = x.std(axis=0, dtype=np.float64).astype(np.float32) if standardize else np.ones(x.shape[1], dtype=np.float32)
        scale = np.where(scale > 0, scale, 1).astype(np.float32)
        self.register_buffer("feature_mean", torch.from_numpy(mean))
        self.register_buffer("feature_scale", torch.from_numpy(scale))
        self.head = nn.Linear(x.shape[1], outputs)

    def get_extra_state(self):
        return {"task": self.task, "classes": self.classes}

    def set_extra_state(self, state):
        if state["task"] != self.task:
            raise ValueError("probe checkpoint task mismatch")
        self.classes = tuple(state["classes"])

    def encode_targets(self, targets: np.ndarray) -> torch.Tensor:
        values = np.asarray(targets)
        if values.ndim != 1 or not len(values):
            raise ValueError("targets must be a non-empty vector")
        if self.task == "regression":
            encoded = values.astype(np.float32)
            if not np.isfinite(encoded).all():
                raise ValueError("regression targets must be finite")
            return torch.from_numpy(encoded)
        vocabulary = {value: index for index, value in enumerate(self.classes)}
        try:
            return torch.tensor([vocabulary[value] for value in values.tolist()], dtype=torch.long)
        except (KeyError, TypeError) as error:
            raise ValueError("evaluation target contains a class absent from training") from error

    def forward(self, batch):
        output = self.head((batch["features"] - self.feature_mean) / self.feature_scale)
        return output if self.task == "classification" else output.squeeze(-1)

    def compute_loss(self, output, batch) -> LossOutput:
        loss = F.cross_entropy(output, batch["targets"]) if self.task == "classification" else F.mse_loss(output, batch["targets"])
        return LossOutput(loss, {"prediction_loss": loss})

    def on_optimizer_step(self) -> None:
        pass


def feature_batches(probe: LinearProbe, features: np.ndarray, targets: np.ndarray, *, batch_size: int, device="cpu") -> list[dict[str, torch.Tensor]]:
    """Fixed-order batches; no independent loader RNG or encoder execution."""
    values = np.asarray(features, dtype=np.float32)
    if batch_size <= 0 or values.ndim != 2 or values.shape[1] != probe.head.in_features or len(values) != len(targets) or not np.isfinite(values).all():
        raise ValueError("invalid or unaligned feature batches")
    x = torch.from_numpy(values.copy())
    y = probe.encode_targets(targets)
    return [{"features": x[start:start + batch_size].to(device), "targets": y[start:start + batch_size].to(device)}
            for start in range(0, len(x), batch_size)]


@torch.no_grad()
def evaluate_probe(probe: LinearProbe, batches: Iterable[dict[str, torch.Tensor]], *, metrics: tuple[str, ...]) -> dict[str, float]:
    if not metrics or any(metric not in SUPPORTED_METRICS[probe.task] for metric in metrics):
        raise ValueError("unsupported probe metrics")
    probe.eval()
    predictions, targets = [], []
    for batch in batches:
        output = probe(batch)
        predictions.append((output.argmax(dim=-1) if probe.task == "classification" else output).cpu().numpy())
        targets.append(batch["targets"].cpu().numpy())
    if not predictions:
        raise ValueError("evaluation batches are empty")
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("evaluation produced non-finite values")
    result = {}
    for metric in metrics:
        if metric == "accuracy":
            value = np.mean(prediction == target)
        elif metric == "macro_f1":
            scores = []
            # Fixed training vocabulary, including absent evaluation classes.
            for index in range(len(probe.classes)):
                tp = np.sum((prediction == index) & (target == index))
                fp = np.sum((prediction == index) & (target != index))
                fn = np.sum((prediction != index) & (target == index))
                denominator = 2 * tp + fp + fn
                scores.append(2 * tp / denominator if denominator else 0.0)
            value = np.mean(scores)
        elif metric == "mae":
            value = np.mean(np.abs(prediction.astype(np.float64) - target))
        else:  # rmse
            value = np.sqrt(np.mean((prediction.astype(np.float64) - target) ** 2))
        result[metric] = float(value)
    return result
