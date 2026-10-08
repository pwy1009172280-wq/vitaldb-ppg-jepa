"""One explicit frozen, layer-wise probe entry point; no SSL method branches."""

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path

import numpy as np

from ...data.access import SplitContext
from ...data.leakage import assert_manifest_splits_disjoint, assert_pretraining_downstream_disjoint
from ...data.splits import SubjectSplit
from ...experiments.metadata import save_json_metadata, seed_everything
from ...experiments.protocol import EvaluationProtocol
from ...experiments.run import create_run
from ...training.checkpoint import CheckpointManager, CheckpointSelectionPolicy
from ...training.metrics import MetricsLogger
from ...training.optim import OptimizerFactory
from ...training.trainer import Trainer, TrainerConfig
from .config import LayerProbeConfig
from .features import (
    EncoderFactory, FeatureBatch, RepresentationReader, extract_layer_features,
    load_frozen_encoder, read_encoder_depths, save_extracted_features,
)
from .probes import LinearProbe, evaluate_probe, feature_batches


def run_layer_probe(
    config: LayerProbeConfig,
    loaders: Mapping[str, Iterable[FeatureBatch]],
    *,
    encoder_factory: EncoderFactory,
    optimizer_factory: OptimizerFactory,
    split: SubjectSplit,
    dataset_manifest_ref: str,
    subject_split_ref: str,
    evaluation_protocol: EvaluationProtocol,
    pretraining_subject_ids: tuple[str, ...],
    results_root: str | Path = "results",
    device="cpu",
    representation_reader: RepresentationReader = read_encoder_depths,
    save_features: bool = True,
) -> dict:
    """Extract each split once, train one head per layer, record all splits.

    The caller supplies already split/preprocessed loaders and reconstructs the
    existing encoder architecture from its checkpoint. No dataset is opened by
    this module. This phase uses last-head selection, not test-set selection.
    """
    roles = ("train", "validation", "test")
    if set(loaders) != set(roles):
        raise ValueError("loaders must declare train, validation, and test")
    assert_manifest_splits_disjoint(split.as_dict())
    assert_pretraining_downstream_disjoint(pretraining_subject_ids, split.validation | split.test)
    if len(set(pretraining_subject_ids)) != len(pretraining_subject_ids):
        raise ValueError("pretraining subject IDs must be unique")
    if (
        evaluation_protocol.task != config.downstream.task
        or evaluation_protocol.protocol_type != "linear_probe"
        or not evaluation_protocol.backbone_frozen
        or evaluation_protocol.aggregation_level != "sample"
        or evaluation_protocol.metrics != (config.downstream.metric,)
    ):
        raise ValueError("evaluation protocol must match the frozen sample-level layer-probe config")
    if not Path(config.encoder.checkpoint).is_file():
        raise FileNotFoundError(config.encoder.checkpoint)
    # Freeze/load and validate extraction before initializing an output run.
    seed_everything(config.experiment.seed)
    checkpoint_reference = str(Path(config.encoder.checkpoint).resolve())
    encoder = load_frozen_encoder(checkpoint_reference, encoder_factory, device=device)
    bundles = {
        role: extract_layer_features(
            encoder, loaders[role], layers=config.encoder.layers,
            context=SplitContext.from_split(split, role), checkpoint_reference=checkpoint_reference,
            device=device, representation_reader=representation_reader,
        ) for role in roles
    }
    assert_manifest_splits_disjoint({role: bundle.subject_ids for role, bundle in bundles.items()})
    targets = {}
    for role, bundle in bundles.items():
        try:
            targets[role] = np.asarray([row["labels"][config.downstream.target] for row in bundle.samples])
        except KeyError as error:
            raise ValueError(f"{role} samples are missing target {config.downstream.target!r}") from error
    # Resolve paths once so checkpoint/metrics references remain unambiguous.
    resolved = config.to_dict()
    resolved["encoder"]["checkpoint"] = checkpoint_reference
    run, manifest = create_run(
        config.experiment.name, resolved, dataset_manifest_ref, subject_split_ref,
        config.experiment.seed, results_root=results_root, evaluation_protocol=evaluation_protocol,
        pretraining_subject_ids=pretraining_subject_ids,
    )
    protocol_path = run.path.resolve() / "evaluation_protocol.json"
    save_json_metadata(evaluation_protocol, protocol_path)
    protocol_digest = hashlib.sha256(json.dumps(evaluation_protocol.to_dict(), sort_keys=True).encode()).hexdigest()
    if save_features:
        for role, bundle in bundles.items():
            save_extracted_features(bundle, run.path / "features" / f"{role}.npz")
    rows = []
    for layer in config.encoder.layers:
        # Same seed per head: equal-width layers receive the same initialization.
        seed_everything(config.experiment.seed)
        probe = LinearProbe(
            bundles["train"].features[layer], targets["train"], task=config.downstream.task,
            standardize=config.probe.standardize,
        ).to(device)
        batches = {
            role: feature_batches(probe, bundle.features[layer], targets[role], batch_size=config.probe.batch_size, device=device)
            for role, bundle in bundles.items()
        }
        optimizer = optimizer_factory.create(
            config.probe.optimizer["name"], probe.parameters(), **config.probe.optimizer["kwargs"],
        )
        manager = CheckpointManager(run.checkpoints_path / layer, CheckpointSelectionPolicy(config.probe.checkpoint_selection))
        head_config = dict(resolved, selected_layer=layer)
        trainer = Trainer(
            probe, optimizer, config=TrainerConfig(amp=False, scheduler_step_policy="none"),
            checkpoint_manager=manager,
            metrics_logger=MetricsLogger(run.logs_path / f"{layer}.json", protocol_reference=str(protocol_path)),
            protocol_reference=str(protocol_path), experiment_manifest_reference=str(run.manifest_path.resolve()),
            resolved_config=head_config,
        )
        trainer.fit(batches["train"], validation_loader=batches["validation"], epochs=config.probe.epochs)
        for role in roles:
            rows.append({
                "layer_name": layer, "task": config.downstream.task, "target": config.downstream.target,
                "split": role, "metrics": evaluate_probe(probe, batches[role], metrics=evaluation_protocol.metrics),
                "feature_readout": bundles[role].pooling[layer],
                "aggregation": {"level": "sample", "num_samples": len(bundles[role].samples), "num_subjects": len(bundles[role].subject_ids)},
                "checkpoint_reference": resolved["encoder"]["checkpoint"],
                "probe_checkpoint_reference": str((manager.directory / "last.pt").resolve()),
                "protocol_reference": str(protocol_path), "protocol_hash": protocol_digest,
            })
    result = {
        "experiment_name": manifest.experiment_name,
        "experiment_manifest_reference": str(run.manifest_path.resolve()),
        "checkpoint_selection": config.probe.checkpoint_selection,
        "evaluation_protocol": evaluation_protocol.to_dict(),
        "rows": rows,
    }
    save_json_metadata(result, run.metrics_path)
    return result
