"""All fixtures/checkpoints are synthetic; never open a stored dataset."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import SplitAwareDataset, SplitContext, SubjectLeakageError, SubjectSplit, UnifiedSample
from src.data.base import BaseDataset
from src.downstream.layer_probe import (
    DownstreamConfig, EncoderConfig, ExperimentConfig, LayerProbeConfig,
    LinearProbe, ProbeConfig, collate_feature_samples, evaluate_probe, extract_layer_features,
    feature_batches, load_extracted_features, load_frozen_encoder, load_layer_probe_config,
    run_layer_probe, save_extracted_features,
)
from src.experiments import EvaluationProtocol
from src.experiments.metadata import save_config
from src.training import OptimizerFactory


class SyntheticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 4)
        self.norm = nn.LayerNorm(4)
        self.dropout = nn.Dropout(0.8)
        self.register_buffer("version", torch.tensor(1))

    def encode_full(self, signal):
        shallow = self.projection(signal.transpose(1, 2))
        middle = self.dropout(torch.tanh(shallow))
        return SimpleNamespace(hidden_states=(shallow, middle), final_tokens=self.norm(middle))


class SyntheticDataset(BaseDataset):
    def __init__(self, samples):
        self.samples = samples

    @property
    def name(self):
        return "synthetic"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def subject_ids(self):
        return frozenset(sample.subject_id for sample in self.samples)


def sample(subject, index, value):
    return UnifiedSample(
        signal=np.full((1, 8), value, dtype=np.float32), subject_id=subject,
        recording_id=f"{subject}-recording", dataset="synthetic", modality="synthetic",
        sampling_rate_hz=8, start_time_s=float(index), end_time_s=float(index + 1),
        window_id=f"window-{index}", window_start_sample=index * 8, window_end_sample=(index + 1) * 8,
        channel_names=("channel",), units=("a.u.",),
        provenance={"source": "synthetic-fixture"}, metadata={"session": "synthetic-session"},
        labels={"category": "positive" if value > 0 else "negative", "value": np.float32(value * 2)},
    )


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(17)
    model = SyntheticEncoder()
    path = tmp_path / "encoder.pt"
    torch.save({"model": model.state_dict(), "resolved_config": {"encoder": "synthetic"}}, path)
    return path


@pytest.fixture
def split_loaders():
    split = SubjectSplit(frozenset({"train-subject"}), frozenset({"validation-subject"}), frozenset({"test-subject"}))
    loaders = {}
    for role in ("train", "validation", "test"):
        subject = next(iter(getattr(split, role)))
        samples = [sample(subject, index, value) for index, value in enumerate((-2., -1., 1., 2.))]
        view = SplitAwareDataset(SyntheticDataset(samples), SplitContext.from_split(split, role))
        loaders[role] = DataLoader(view, batch_size=3, collate_fn=collate_feature_samples)
    return split, loaders


def config_for(checkpoint, task="classification"):
    return LayerProbeConfig(
        ExperimentConfig(f"synthetic-{task}", seed=3), EncoderConfig(str(checkpoint), ("layer_1", "layer_2", "final_layer")),
        DownstreamConfig(task, "category" if task == "classification" else "value", "accuracy" if task == "classification" else "mae"),
        ProbeConfig(epochs=2, batch_size=3, optimizer={"name": "sgd", "kwargs": {"lr": 0.1}}),
    )


def run_synthetic(config, split, loaders, root, **kwargs):
    factory = OptimizerFactory()
    factory.register("sgd", torch.optim.SGD)
    protocol = EvaluationProtocol(config.downstream.task, "linear_probe", (config.downstream.metric,), "sample", True)
    return run_layer_probe(
        config, loaders, encoder_factory=lambda payload: SyntheticEncoder(), optimizer_factory=factory,
        split=split, dataset_manifest_ref="manifest://synthetic-dataset", subject_split_ref="split://synthetic",
        evaluation_protocol=kwargs.pop("evaluation_protocol", protocol),
        pretraining_subject_ids=kwargs.pop("pretraining_subject_ids", ("source-pretrain-subject",)),
        results_root=root, **kwargs,
    )


def test_config_round_trip_and_example(tmp_path, checkpoint):
    config = config_for(checkpoint)
    path = save_config(config, tmp_path / "config.yaml")
    assert load_layer_probe_config(path) == config
    example = Path(__file__).resolve().parents[1] / "configs" / "layer_probe.yaml"
    assert load_layer_probe_config(example).experiment.type == "layer_probe"
    with pytest.raises(ValueError, match="unknown"):
        LayerProbeConfig.from_dict(dict(config.to_dict(), fine_tuning={}))
    with pytest.raises(ValueError, match="last"):
        replace(config.probe, checkpoint_selection="best")


def test_frozen_extraction_and_cache_preserve_provenance(tmp_path, checkpoint, split_loaders):
    split, loaders = split_loaders
    encoder = load_frozen_encoder(checkpoint, lambda payload: SyntheticEncoder())
    before = {key: value.clone() for key, value in encoder.state_dict().items()}
    bundle = extract_layer_features(
        encoder, loaders["train"], layers=("layer_1", "layer_2", "final_layer"),
        context=SplitContext.from_split(split, "train"), checkpoint_reference=str(checkpoint),
    )
    assert all(matrix.shape == (4, 4) for matrix in bundle.features.values())
    assert all(torch.equal(before[key], tensor) for key, tensor in encoder.state_dict().items())
    assert not encoder.training and all(not parameter.requires_grad and parameter.grad is None for parameter in encoder.parameters())
    again = extract_layer_features(
        encoder, loaders["train"], layers=("layer_1",),
        context=SplitContext.from_split(split, "train"), checkpoint_reference=str(checkpoint),
    )
    assert np.array_equal(bundle.features["layer_1"], again.features["layer_1"])
    path = save_extracted_features(bundle, tmp_path / "features.custom_suffix")
    loaded = load_extracted_features(path)
    assert loaded.samples == bundle.samples
    assert loaded.checkpoint_reference == str(checkpoint) and loaded.split == "train"
    assert loaded.pooling == {name: "mean_tokens" for name in bundle.features}
    assert loaded.samples[0]["subject_id"] == "train-subject"
    assert loaded.samples[0]["window_start_sample"] == 0
    assert loaded.samples[0]["provenance"] == {"source": "synthetic-fixture"}
    assert all(np.array_equal(loaded.features[name], bundle.features[name]) for name in bundle.features)


def test_extraction_requires_freezing_and_valid_layers(checkpoint, split_loaders):
    split, loaders = split_loaders
    kwargs = dict(context=SplitContext.from_split(split, "train"), checkpoint_reference=str(checkpoint))
    with pytest.raises(ValueError, match="frozen"):
        extract_layer_features(SyntheticEncoder(), loaders["train"], layers=("layer_1",), **kwargs)
    encoder = load_frozen_encoder(checkpoint, lambda payload: SyntheticEncoder())
    with pytest.raises(ValueError, match="requested layer"):
        extract_layer_features(encoder, loaders["train"], layers=("layer_9",), **kwargs)
    with pytest.raises(ValueError, match="unique"):
        extract_layer_features(encoder, loaders["train"], layers=("layer_1", "layer_1"), **kwargs)
    with pytest.raises(ValueError, match="layer_1"):
        extract_layer_features(encoder, loaders["train"], layers=("block_1",), **kwargs)


def test_extraction_rejects_wrong_subject_empty_and_invalid_mask(checkpoint, split_loaders):
    split, loaders = split_loaders
    encoder = load_frozen_encoder(checkpoint, lambda payload: SyntheticEncoder())
    kwargs = dict(layers=("layer_1",), context=SplitContext.from_split(split, "train"), checkpoint_reference=str(checkpoint))
    with pytest.raises(SubjectLeakageError):
        extract_layer_features(encoder, loaders["validation"], **kwargs)
    with pytest.raises(ValueError, match="empty"):
        extract_layer_features(encoder, [], **kwargs)
    invalid = replace(sample("train-subject", 0, 1), valid_mask=np.zeros(8, dtype=bool))
    with pytest.raises(ValueError, match="invalid/padded"):
        extract_layer_features(encoder, [collate_feature_samples([invalid])], **kwargs)
    duplicate = collate_feature_samples([sample("train-subject", 0, 1)] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        extract_layer_features(encoder, [duplicate], **kwargs)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_complete_layer_probe_run_and_existing_checkpoint_contract(tmp_path, checkpoint, split_loaders, task):
    split, loaders = split_loaders
    config = config_for(checkpoint, task)
    result = run_synthetic(config, split, loaders, tmp_path / "results")
    run = tmp_path / "results" / config.experiment.name
    assert len(result["rows"]) == 9
    assert {(row["layer_name"], row["split"]) for row in result["rows"]} == {
        (layer, role) for layer in config.encoder.layers for role in ("train", "validation", "test")
    }
    assert json.loads((run / "metrics.json").read_text()) == result
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["evaluation_protocol"]["backbone_frozen"] is True
    assert manifest["dataset_manifest_ref"] == "manifest://synthetic-dataset"
    assert manifest["pretraining_subject_ids"] == ["source-pretrain-subject"]
    for role in ("train", "validation", "test"):
        assert load_extracted_features(run / "features" / f"{role}.npz").split == role
    for row in result["rows"]:
        assert np.isfinite(row["metrics"][config.downstream.metric])
        assert row["aggregation"]["num_samples"] == 4
        assert Path(row["checkpoint_reference"]).resolve() == checkpoint.resolve()
        assert Path(row["protocol_reference"]).is_file() and row["protocol_hash"]
        payload = torch.load(row["probe_checkpoint_reference"], weights_only=False)
        assert payload["epoch"] == 1
        assert payload["resolved_config"]["selected_layer"] == row["layer_name"]
        assert payload["evaluation_protocol_reference"] == row["protocol_reference"]
        assert payload["experiment_manifest_reference"] == str((run / "manifest.json").resolve())
        assert all("encoder" not in key for key in payload["model"])
        logs = json.loads((run / "logs" / f"{row['layer_name']}.json").read_text())
        assert len(logs) == 4 and logs[0]["protocol_reference"] == row["protocol_reference"]
    with pytest.raises(FileExistsError):
        run_synthetic(config, split, loaders, tmp_path / "results")


def test_run_validation_rejects_protocol_and_pretraining_leakage(tmp_path, checkpoint, split_loaders):
    split, loaders = split_loaders
    config = config_for(checkpoint)
    bad_protocol = EvaluationProtocol("classification", "fine_tune", ("accuracy",), "sample", False)
    with pytest.raises(ValueError, match="protocol"):
        run_synthetic(config, split, loaders, tmp_path / "results", evaluation_protocol=bad_protocol)
    with pytest.raises(SubjectLeakageError, match="pretraining"):
        run_synthetic(config, split, loaders, tmp_path / "results", pretraining_subject_ids=("test-subject",))
    with pytest.raises(SubjectLeakageError):
        SubjectSplit(frozenset({"same"}), frozenset({"same"}), frozenset({"other"}))
    bad_loaders = dict(loaders, validation=loaders["train"])
    with pytest.raises(SubjectLeakageError):
        run_synthetic(config, split, bad_loaders, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_train_only_standardization_and_class_vocabulary():
    features = np.array([[-1., 4.], [1., 4.]], dtype=np.float32)
    labels = np.array(["negative", "positive"])
    probe = LinearProbe(features, labels, task="classification")
    mean = probe.feature_mean.clone()
    feature_batches(probe, features + 100, labels, batch_size=2)
    assert torch.equal(mean, probe.feature_mean) and torch.equal(mean, torch.tensor([0., 4.]))
    assert torch.equal(probe.feature_scale, torch.tensor([1., 1.]))
    with pytest.raises(ValueError, match="absent from training"):
        probe.encode_targets(np.array(["unseen"]))
    assert set(dict(probe.named_parameters())) == {"head.weight", "head.bias"}
    assert "feature_mean" in probe.state_dict() and "_extra_state" in probe.state_dict()


@pytest.mark.parametrize("task,metrics", [("classification", ("accuracy", "macro_f1")), ("regression", ("mae", "rmse"))])
def test_metric_values_and_probe_state_round_trip(task, metrics):
    features = np.array([[-1.], [1.]], dtype=np.float32)
    labels = np.array(["a", "b"]) if task == "classification" else np.array([-2., 2.])
    probe = LinearProbe(features, labels, task=task, standardize=False)
    with torch.no_grad():
        probe.head.weight.copy_(torch.tensor([[-1.], [1.]]) if task == "classification" else torch.tensor([[2.]]))
        probe.head.bias.zero_()
    expected = {metric: 1. if task == "classification" else 0. for metric in metrics}
    batches = feature_batches(probe, features, labels, batch_size=1)
    assert evaluate_probe(probe, batches, metrics=metrics) == expected
    restored = LinearProbe(features, labels, task=task)
    restored.load_state_dict(probe.state_dict())
    assert evaluate_probe(restored, batches, metrics=metrics) == expected


def test_existing_standard_jepa_encode_full_is_supported(tmp_path):
    # Existing model only; this test adds no SSL implementation or adapter.
    from src.models.common import BackboneConfig
    from src.models.jepa import JEPA1D
    factory = lambda payload: JEPA1D(BackboneConfig(8, 2, 2, 2., 0.), patch_size=4, predictor_depth=1, predictor_num_heads=2)
    source = factory({})
    checkpoint = tmp_path / "jepa.pt"
    torch.save({"model": source.state_dict(), "resolved_config": {"synthetic": True}}, checkpoint)
    encoder = load_frozen_encoder(checkpoint, factory)
    batch = collate_feature_samples([sample("subject", 0, 1.)])
    expected = encoder.encode_full(batch.signal)
    bundle = extract_layer_features(
        encoder, [batch], layers=("layer_1", "layer_2", "final_layer"),
        context=SplitContext("test", frozenset({"subject"})), checkpoint_reference=str(checkpoint),
    )
    assert np.allclose(bundle.features["layer_1"], expected.hidden_states[0].mean(1).numpy())
    assert np.allclose(bundle.features["final_layer"], expected.final_tokens.mean(1).numpy())


def test_injected_reader_and_optional_cache(tmp_path, checkpoint, split_loaders):
    split, loaders = split_loaders
    calls = []

    def reader(encoder, batch):
        calls.append(len(batch.samples))
        values = encoder.encode_full(batch.signal)
        return {"layer_1": values.hidden_states[0].mean(1)}

    config = config_for(checkpoint)
    config = replace(config, encoder=replace(config.encoder, layers=("layer_1",)))
    result = run_synthetic(config, split, loaders, tmp_path / "results", representation_reader=reader, save_features=False)
    assert calls == [3, 1, 3, 1, 3, 1]
    assert len(result["rows"]) == 3
    assert all(row["feature_readout"] == "reader_prepooled" for row in result["rows"])
    assert not (tmp_path / "results" / config.experiment.name / "features").exists()


def test_invalid_checkpoint_and_missing_targets_fail(tmp_path, checkpoint, split_loaders):
    missing_state = tmp_path / "missing-state.pt"
    torch.save({"resolved_config": {}}, missing_state)
    with pytest.raises(ValueError, match="state dictionary"):
        load_frozen_encoder(missing_state, lambda payload: SyntheticEncoder())
    with pytest.raises(RuntimeError):
        load_frozen_encoder(checkpoint, lambda payload: nn.Linear(1, 2))
    split, loaders = split_loaders
    config = config_for(checkpoint)
    config = replace(config, downstream=replace(config.downstream, target="missing"))
    with pytest.raises(ValueError, match="missing target"):
        run_synthetic(config, split, loaders, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_import_does_not_load_concrete_dataset_or_ssl_models():
    script = (
        "import sys; import src.downstream.layer_probe; "
        "assert 'src.data.processed_dataset' not in sys.modules; "
        "assert not any(name.startswith('src.models.') for name in sys.modules); "
        "assert 'src.downstream.layerwise' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[1])
