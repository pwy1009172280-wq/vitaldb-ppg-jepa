"""Generic probe components independent of extracted feature storage."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...training.contracts import LossOutput


class FeatureNormalizer:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, features: np.ndarray, *, split_role: str) -> "FeatureNormalizer":
        if split_role != "train":
            raise ValueError("FeatureNormalizer may only be fit on the train split")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("normalizer fit requires finite [samples, features]")
        self.mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        self.scale = np.where(self.scale > 0, self.scale, 1).astype(np.float32)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("FeatureNormalizer must be fit before transform")
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("feature dimensions do not match normalizer")
        return ((values - self.mean) / self.scale).astype(np.float32)


class ClassVocabulary:
    def __init__(self, classes: tuple[Any, ...] = ()) -> None:
        self.classes = tuple(classes)

    def fit(self, labels: np.ndarray, *, split_role: str) -> "ClassVocabulary":
        if split_role != "train":
            raise ValueError("ClassVocabulary may only be fit on the train split")
        values = np.asarray(labels)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("classification labels must be a non-empty vector")
        self.classes = tuple(np.unique(values).tolist())
        if len(self.classes) < 2:
            raise ValueError("classification training requires at least two classes")
        return self

    def encode(self, labels: np.ndarray) -> np.ndarray:
        lookup = {value: index for index, value in enumerate(self.classes)}
        try:
            return np.asarray([lookup[value] for value in np.asarray(labels).tolist()], dtype=np.int64)
        except KeyError as error:
            raise ValueError("evaluation labels contain a class absent from train vocabulary") from error


class LinearHead(nn.Module):
    """Pure linear head; it has no knowledge of datasets or feature extraction."""

    def __init__(self, input_dim: int, output_dim: int, *, task: str) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or task not in ("classification", "regression"):
            raise ValueError("invalid linear head configuration")
        self.task = task
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("LinearHead expects [samples, features]")
        output = self.linear(features)
        return output.squeeze(-1) if self.task == "regression" else output


class ProbeModel(nn.Module):
    """Trainer adapter around a pure LinearHead and train-fitted transforms."""

    def __init__(self, head: LinearHead, normalizer: FeatureNormalizer, vocabulary: ClassVocabulary | None = None) -> None:
        super().__init__()
        if normalizer.mean is None or normalizer.scale is None:
            raise RuntimeError("ProbeModel requires a fitted FeatureNormalizer")
        self.head = head
        self.normalizer = normalizer
        self.vocabulary = vocabulary
        self.register_buffer("feature_mean", torch.from_numpy(normalizer.mean.copy()))
        self.register_buffer("feature_scale", torch.from_numpy(normalizer.scale.copy()))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = (batch["features"] - self.feature_mean) / self.feature_scale
        return self.head(features)

    def compute_loss(self, output: torch.Tensor, batch: dict[str, torch.Tensor]) -> LossOutput:
        if self.head.task == "classification":
            loss = F.cross_entropy(output, batch["targets"])
        else:
            loss = F.mse_loss(output, batch["targets"])
        return LossOutput(loss, {"prediction_loss": loss})

    def on_optimizer_step(self) -> None:
        pass


@dataclass(frozen=True)
class PredictionBatch:
    targets: np.ndarray
    predicted_labels: np.ndarray | None = None
    scores: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    class_vocabulary: tuple[Any, ...] = ()
    task: str = ""
    subject_ids: tuple[str, ...] = ()


def make_prediction_batch(
    head: LinearHead,
    features: np.ndarray,
    labels: np.ndarray,
    normalizer: FeatureNormalizer,
    vocabulary: ClassVocabulary | None = None,
    *,
    subject_ids: tuple[str, ...] = (),
) -> PredictionBatch:
    values = torch.from_numpy(normalizer.transform(features))
    with torch.no_grad():
        output = head(values).detach().cpu().numpy()
    targets = np.asarray(labels)
    if head.task == "regression":
        return PredictionBatch(targets, scores=output.astype(np.float64), task=head.task, subject_ids=subject_ids)
    if vocabulary is None:
        raise ValueError("classification prediction requires ClassVocabulary")
    logits = np.asarray(output, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1).astype(np.int64)
    encoded = vocabulary.encode(targets)
    return PredictionBatch(encoded, predicted, scores=logits, probabilities=probabilities,
                           class_vocabulary=vocabulary.classes, task=head.task, subject_ids=subject_ids)
