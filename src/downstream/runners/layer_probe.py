"""Layer-probe orchestration over the common downstream core."""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...experiments.metadata import save_json_metadata, seed_everything
from ...experiments.protocol import EvaluationProtocol
from ...experiments.run import create_run
from ...training.checkpoint import CheckpointManager, CheckpointSelectionPolicy
from ...training.metrics import MetricsLogger
from ...training.optim import OptimizerFactory
from ...training.trainer import Trainer, TrainerConfig
from ...data.leakage import assert_pretraining_downstream_disjoint
from ..core.aggregate import aggregate_predictions
from ..core.metrics import evaluate_predictions
from ..core.probe import (
    ClassVocabulary,
    FeatureNormalizer,
    LinearHead,
    PredictionBatch,
    ProbeModel,
    make_prediction_batch,
)
from ..core.protocol import DownstreamPolicy
from ..core.results import save_results
from ..core.extraction import ExtractedFeature
from ..core.tasks import LabelTaskAdapter, TaskAdapter


def _labels(feature: ExtractedFeature, adapter: TaskAdapter) -> np.ndarray:
    return np.asarray([adapter.adapt(row).value for row in feature.samples])


def _batches(feature: ExtractedFeature, labels: np.ndarray, normalizer: FeatureNormalizer, vocabulary: ClassVocabulary | None, task: str, batch_size: int, device: str | torch.device):
    values = torch.from_numpy(feature.values.copy())
    if task == "classification":
        targets = torch.from_numpy(vocabulary.encode(labels))
    else:
        targets = torch.from_numpy(labels.astype(np.float32))
    return [
        {"features": values[start:start + batch_size].to(device), "targets": targets[start:start + batch_size].to(device)}
        for start in range(0, len(values), batch_size)
    ]


def _subjects(feature: ExtractedFeature) -> tuple[str, ...]:
    return tuple(str(row["subject_id"]) for row in feature.samples)


def run_layer_probe(
    features: Mapping[str, Mapping[str, ExtractedFeature]],
    *,
    experiment_name: str,
    target: str | None = None,
    task_adapter: TaskAdapter | None = None,
    protocol: EvaluationProtocol,
    policy: DownstreamPolicy,
    optimizer_factory: OptimizerFactory,
    optimizer_name: str = "sgd",
    optimizer_kwargs: dict[str, Any] | None = None,
    epochs: int = 1,
    batch_size: int = 32,
    seed: int = 0,
    dataset_manifest_ref: str,
    subject_split_ref: str,
    results_root: str | Path = "results",
    resolved_config: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    pretraining_subject_ids: tuple[str, ...] = (),
    overwrite_run: bool = False,
) -> dict[str, Any]:
    """Train one simple head per representation and write versioned results.

    Feature extraction/cache is intentionally separate: this runner consumes
    provenance-preserving ``ExtractedFeature`` objects produced by core.
    """
    if task_adapter is None:
        if not target:
            raise ValueError("an explicit target or task_adapter is required")
        task_adapter = LabelTaskAdapter(target)
    if protocol.protocol_type != "linear_probe" or not protocol.backbone_frozen:
        raise ValueError("layer probing requires the authoritative frozen linear_probe protocol")
    if policy.checkpoint_selection == "best_validation" and policy.selection_metric not in protocol.metrics:
        raise ValueError("checkpoint selection metric must be listed in EvaluationProtocol.metrics")
    if set(features) != {"train", "validation", "test"}:
        raise ValueError("features must contain train, validation, and test")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    split_subjects = {
        role: set(_subjects(next(iter(feature_map.values()))))
        for role, feature_map in features.items()
        if feature_map
    }
    if split_subjects["train"] & split_subjects["validation"] or split_subjects["train"] & split_subjects["test"] or split_subjects["validation"] & split_subjects["test"]:
        raise ValueError("subject overlap across downstream splits")
    assert_pretraining_downstream_disjoint(pretraining_subject_ids, split_subjects["validation"] | split_subjects["test"])
    seed_everything(seed)
    resolved = dict(resolved_config or {})
    resolved.update({"task_adapter": task_adapter.to_dict(), "task": protocol.task, "representations": sorted(set().union(*(set(value) for value in features.values())))})
    run, manifest = create_run(
        experiment_name, resolved, dataset_manifest_ref, subject_split_ref, seed,
        results_root=results_root, evaluation_protocol=protocol,
        overwrite=overwrite_run,
    )
    manifest_payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    manifest_payload.update({"schema_version": 1, "downstream_policy": policy.to_dict()})
    save_json_metadata(manifest_payload, run.manifest_path)
    protocol_path = run.path / "evaluation_protocol.json"
    save_json_metadata({"schema_version": 1, **protocol.to_dict()}, protocol_path)

    rows = []
    prediction_dir = run.path / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for representation in sorted(set().union(*(set(value) for value in features.values()))):
        train_feature = features["train"][representation]
        normalizer = FeatureNormalizer().fit(train_feature.values, split_role="train")
        train_labels = _labels(train_feature, task_adapter)
        vocabulary = ClassVocabulary().fit(train_labels, split_role="train") if protocol.task == "classification" else None
        output_dim = len(vocabulary.classes) if vocabulary is not None else 1
        head = LinearHead(train_feature.values.shape[1], output_dim, task=protocol.task).to(device)
        model = ProbeModel(head, normalizer, vocabulary).to(device)
        optimizer = optimizer_factory.create(optimizer_name, model.parameters(), **(optimizer_kwargs or {"lr": 0.01}))
        checkpoint_policy = CheckpointSelectionPolicy(
            "last" if policy.checkpoint_selection == "last" else "monitored_metric",
            policy.selection_metric if policy.checkpoint_selection == "best_validation" else None,
            policy.selection_mode if policy.checkpoint_selection == "best_validation" else None,
        )
        manager = CheckpointManager(run.checkpoints_path / representation, checkpoint_policy)
        protocol_reference = str(protocol_path.resolve())
        trainer = Trainer(
            model, optimizer, config=TrainerConfig(scheduler_step_policy="none"),
            metrics_logger=MetricsLogger(run.logs_path / f"{representation}.json", protocol_reference),
            protocol_reference=protocol_reference,
            experiment_manifest_reference=str(run.manifest_path.resolve()),
            resolved_config=dict(resolved, representation=representation),
        )
        training_batches = _batches(train_feature, train_labels, normalizer, vocabulary, protocol.task, batch_size, device)
        best_value: float | None = None
        for _ in range(epochs):
            trainer.fit(training_batches, epochs=1)
            validation = features["validation"][representation]
            validation_prediction = make_prediction_batch(
                head, validation.values, _labels(validation, task_adapter), normalizer, vocabulary,
                subject_ids=_subjects(validation),
            )
            validation_metrics = evaluate_predictions(validation_prediction, protocol.metrics)
            manager.save(
                "last", model, optimizer, None, trainer.scaler,
                epoch=trainer.epoch - 1, global_step=trainer.global_step,
                best_metric_name=policy.selection_metric if policy.checkpoint_selection == "best_validation" else None,
                best_metric_value=validation_metrics.get(policy.selection_metric) if policy.selection_metric else None,
                experiment_manifest_reference=str(run.manifest_path.resolve()),
                evaluation_protocol_reference=protocol_reference,
                resolved_config=dict(resolved, representation=representation),
            )
            if policy.checkpoint_selection == "best_validation":
                candidate = validation_metrics.get(policy.selection_metric)
                if candidate is None:
                    raise ValueError("selection metric is undefined on validation split")
                better = best_value is None or (candidate < best_value if policy.selection_mode == "min" else candidate > best_value)
                if better:
                    best_value = candidate
                    manager.save_best(
                        model, optimizer, None, trainer.scaler,
                        epoch=trainer.epoch - 1, global_step=trainer.global_step,
                        best_metric_name=policy.selection_metric, best_metric_value=candidate,
                        experiment_manifest_reference=str(run.manifest_path.resolve()),
                        evaluation_protocol_reference=protocol_reference,
                        resolved_config=dict(resolved, representation=representation),
                    )
        selected = "best" if policy.checkpoint_selection == "best_validation" else "last"
        manager.load(
            selected, model, optimizer, None, trainer.scaler,
            expected_resolved_config=dict(resolved, representation=representation),
            expected_experiment_manifest_reference=str(run.manifest_path.resolve()),
            expected_evaluation_protocol_reference=protocol_reference,
        )
        for role in ("train", "validation", "test"):
            feature = features[role][representation]
            prediction = make_prediction_batch(
                head, feature.values, _labels(feature, task_adapter), normalizer, vocabulary,
                subject_ids=_subjects(feature),
            )
            prediction = aggregate_predictions(prediction, protocol.aggregation_level)
            prediction_path = prediction_dir / f"{representation}__{role}.npz"
            np.savez_compressed(
                prediction_path,
                schema_version=np.asarray(1, dtype=np.int64),
                targets=prediction.targets,
                predicted_labels=prediction.predicted_labels if prediction.predicted_labels is not None else np.asarray([]),
                scores=prediction.scores if prediction.scores is not None else np.asarray([]),
                probabilities=prediction.probabilities if prediction.probabilities is not None else np.asarray([]),
                subject_ids=np.asarray(prediction.subject_ids, dtype=str),
                class_vocabulary=np.asarray(prediction.class_vocabulary, dtype=str),
            )
            rows.append({
                "representation": representation,
                "task": protocol.task,
                "split": role,
                "metrics": evaluate_predictions(prediction, protocol.metrics),
                "aggregation": protocol.aggregation_level,
                "num_samples": len(feature.samples),
                "num_subjects": len(set(_subjects(feature))),
                "checkpoint_reference": feature.checkpoint_reference,
                "probe_checkpoint_reference": str((manager.directory / f"{selected}.pt").resolve()),
                "reader_name": feature.reader_name,
                "reader_version": feature.reader_version,
                "pooling": feature.pooling,
                "seed": seed,
                "prediction_path": str(prediction_path.resolve()),
            })
    result = {
        "schema_version": 1,
        "experiment_name": manifest.experiment_name,
        "evaluation_protocol": protocol.to_dict(),
        "downstream_policy": policy.to_dict(),
        "task_adapter": task_adapter.to_dict(),
        "rows": rows,
    }
    save_results(result, run.metrics_path)
    return result
