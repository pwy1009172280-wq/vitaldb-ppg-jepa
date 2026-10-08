"""Synthetic validation for the common downstream layer."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.data import SplitContext, SubjectSplit, UnifiedSample
from src.downstream.core import (
    DownstreamPolicy,
    FeatureBatch,
    FeatureCacheKey,
    FeatureNormalizer,
    LinearHead,
    PredictionBatch,
    Representation,
    aggregate_predictions,
    collate_feature_samples,
    compute_sample_set_hash,
    evaluate_predictions,
    extract_features,
    load_feature_cache,
    save_feature_cache,
)
from src.downstream.readers.jepa import JEPARepresentationReader
from src.downstream.runners.layer_probe import run_layer_probe
from src.experiments import EvaluationProtocol
from src.training import OptimizerFactory


def make_sample(subject, index, label):
    return UnifiedSample(
        signal=np.full((1, 4), float(index + 1), dtype=np.float32),
        subject_id=subject, recording_id=f"recording-{subject}", dataset="synthetic",
        subject_identity_status="RESOLVED", subject_identity_namespace="synthetic",
        modality="synthetic", sampling_rate_hz=1, start_time_s=float(index), end_time_s=float(index + 1),
        window_id=f"window-{subject}-{index}", window_start_sample=index * 4, window_end_sample=(index + 1) * 4,
        labels={"label": label, "value": float(index + 1)}, provenance={"fixture": True},
    )


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 3)


class FakeTransformerReader:
    name = "fake_transformer"
    version = "1"

    def read(self, model, batch):
        token = model.projection(batch.signal.transpose(1, 2))
        return {
            "early_tokens": Representation(token, "token"),
            "late_tokens": Representation(torch.tanh(token), "token"),
        }


class FakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3)


class FakeCNNReader:
    name = "fake_cnn"
    version = "1"

    def read(self, model, batch):
        return {"global_embedding": Representation(model.projection(batch.signal.flatten(1)), "vector")}


def make_loader(subjects_and_labels):
    samples = [make_sample(subject, index, label) for subject, label in subjects_and_labels for index in range(2)]
    return [collate_feature_samples(samples)]


def test_same_core_supports_token_transformer_and_vector_cnn():
    split = SubjectSplit(frozenset({"train"}), frozenset({"validation"}), frozenset({"test"}))
    transformer = extract_features(
        FakeTransformer(), make_loader([("train", "negative")]), reader=FakeTransformerReader(),
        context=SplitContext.from_split(split, "train"), pooling="mean", checkpoint_reference="ckpt",
    )
    cnn = extract_features(
        FakeCNN(), make_loader([("train", "negative")]), reader=FakeCNNReader(),
        context=SplitContext.from_split(split, "train"), pooling="reader_provided", checkpoint_reference="ckpt",
    )
    assert set(transformer) == {"early_tokens", "late_tokens"}
    assert transformer["early_tokens"].values.shape == (2, 3)
    assert set(cnn) == {"global_embedding"}
    assert cnn["global_embedding"].values.shape == (2, 3)


def test_jepa_reader_is_model_specific_but_core_compatible():
    class FakeJEPA(nn.Module):
        def encode_full(self, signal):
            tokens = signal.transpose(1, 2).repeat(1, 1, 3)
            return SimpleNamespace(hidden_states=(tokens, tokens + 1), final_tokens=tokens + 2)

    split = SubjectSplit(frozenset({"train"}), frozenset({"validation"}), frozenset({"test"}))
    reader = JEPARepresentationReader()
    features = extract_features(
        FakeJEPA(), make_loader([("train", "negative")]), reader=reader,
        context=SplitContext.from_split(split, "train"), representations=("layer_1", "final_layer"),
        pooling="mean", checkpoint_reference="ckpt",
    )
    assert set(features) == {"layer_1", "final_layer"}
    assert features["layer_1"].reader_name == reader.name


def test_cache_key_is_strict_and_normalizer_is_train_only(tmp_path):
    split = SubjectSplit(frozenset({"train"}), frozenset({"validation"}), frozenset({"test"}))
    feature = extract_features(
        FakeCNN(), make_loader([("train", "negative")]), reader=FakeCNNReader(),
        context=SplitContext.from_split(split, "train"), pooling="reader_provided", checkpoint_reference="ckpt",
    )["global_embedding"]
    key = FeatureCacheKey(
        "dataset-hash", compute_sample_set_hash(feature.samples), "preprocess-hash", "checkpoint-hash",
        feature.reader_name, feature.reader_version, feature.representation, feature.pooling, "1",
    )
    path = save_feature_cache(feature, tmp_path / "features.npz", key, git_commit="commit")
    loaded = load_feature_cache(path, key)
    assert np.array_equal(loaded.values, feature.values)
    with pytest.raises(ValueError, match="cache key mismatch"):
        load_feature_cache(path, FeatureCacheKey(
            "different-dataset", key.split_sample_set_hash, key.preprocessing_version_hash,
            key.encoder_checkpoint_hash, key.reader_name, key.reader_version, key.representation_name,
            key.pooling_name, key.pooling_version,
        ))
    with pytest.raises(ValueError, match="train split"):
        FeatureNormalizer().fit(feature.values, split_role="validation")

    with pytest.raises(ValueError, match="train split"):
        from src.downstream.core import ClassVocabulary
        ClassVocabulary().fit(np.array([0, 1]), split_role="test")


def test_metrics_require_continuous_scores_and_subject_aggregation():
    prediction = PredictionBatch(
        targets=np.array([0, 0, 1, 1]), predicted_labels=np.array([0, 1, 1, 1]),
        scores=np.array([[-2., -1.], [-1., -0.5], [-0.5, -0.2], [-0.2, 0.3]]),
        probabilities=np.array([[.8, .2], [.4, .6], [.3, .7], [.2, .8]]),
        class_vocabulary=(0, 1), task="classification", subject_ids=("a", "a", "b", "b"),
    )
    metrics = evaluate_predictions(prediction, ("accuracy", "balanced_accuracy", "macro_f1", "AUROC", "AUPRC"))
    assert metrics["AUROC"] is not None and metrics["AUPRC"] is not None
    aggregated = aggregate_predictions(prediction, "mean_probability")
    assert len(aggregated.targets) == 2 and aggregated.probabilities.shape == (2, 2)
    assert aggregate_predictions(prediction, "mean_score").scores.shape == (2, 2)
    inconsistent = PredictionBatch(
        targets=np.array([0, 1]), predicted_labels=np.array([0, 1]), probabilities=np.array([[1., 0.], [0., 1.]]),
        class_vocabulary=(0, 1), task="classification", subject_ids=("same", "same"),
    )
    with pytest.raises(ValueError, match="consistent"):
        aggregate_predictions(inconsistent, "majority_vote")


def test_regression_metrics_keep_original_units():
    prediction = PredictionBatch(
        targets=np.array([10., 20., 30.]), scores=np.array([12., 18., 33.]), task="regression",
    )
    metrics = evaluate_predictions(prediction, ("MAE", "RMSE", "Pearson", "Spearman"))
    assert metrics["MAE"] == pytest.approx(7 / 3)
    assert metrics["Pearson"] is not None and metrics["Spearman"] is not None


def test_end_to_end_core_runner_validation_checkpoint_and_results(tmp_path):
    split = SubjectSplit(frozenset({"train-neg", "train-pos"}), frozenset({"val"}), frozenset({"test"}))
    reader = FakeTransformerReader()
    features = {}
    for role, subjects in {
        "train": [("train-neg", "negative"), ("train-pos", "positive")],
        "validation": [("val", "positive")],
        "test": [("test", "negative")],
    }.items():
        features[role] = extract_features(
            FakeTransformer(), make_loader(subjects), reader=reader,
            context=SplitContext.from_split(split, role), pooling="mean", checkpoint_reference="encoder.pt",
        )
    # Make train/validation/test features share the same representation contract.
    for role in features:
        features[role] = {"early_tokens": features[role]["early_tokens"]}
        feature = features[role]["early_tokens"]
        cache_key = FeatureCacheKey(
            "dataset-hash", compute_sample_set_hash(feature.samples), "preprocess-hash", "checkpoint-hash",
            feature.reader_name, feature.reader_version, feature.representation, feature.pooling, "1",
        )
        cache_path = tmp_path / "cache" / f"{role}.npz"
        save_feature_cache(feature, cache_path, cache_key, git_commit="synthetic-commit")
        features[role] = {"early_tokens": load_feature_cache(cache_path, cache_key)}
    optimizers = OptimizerFactory()
    optimizers.register("sgd", torch.optim.SGD)
    protocol = EvaluationProtocol("classification", "linear_probe", ("accuracy",), "sample", True)
    policy = DownstreamPolicy("fake_transformer", "1", "mean", checkpoint_selection="best_validation", selection_metric="accuracy", selection_mode="max")
    result = run_layer_probe(
        features, experiment_name="synthetic-common-probe", target="label", protocol=protocol,
        policy=policy, optimizer_factory=optimizers, epochs=1, batch_size=2, seed=7,
        dataset_manifest_ref="dataset://synthetic", subject_split_ref="split://synthetic",
        results_root=tmp_path / "results",
    )
    run = tmp_path / "results" / "synthetic-common-probe"
    assert result["schema_version"] == 1
    assert {row["split"] for row in result["rows"]} == {"train", "validation", "test"}
    assert (run / "checkpoints" / "early_tokens" / "best.pt").exists()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["downstream_policy"]["checkpoint_selection"] == "best_validation"
    assert json.loads((run / "metrics.json").read_text())["schema_version"] == 1
