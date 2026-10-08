#!/usr/bin/env python3
"""Authoritative downstream entry point: checkpoint + dataset + reader -> results."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ProcessedPPGUnifiedAdapter, SplitContext, SubjectSplit
from src.data.processed_dataset import ProcessedPPGDataset
from src.downstream.core import FeatureCacheKey, collate_feature_samples, extract_features, save_feature_cache
from src.downstream.core.protocol import DownstreamPolicy
from src.downstream.readers.jepa import JEPARepresentationReader
from src.downstream.runners.layer_probe import run_layer_probe
from src.experiments import EvaluationProtocol
from src.models.factory import build_model
from src.models.jepa import JEPATrainingAdapter
from src.pretrain.config import load_config
from src.training import OptimizerFactory
from src.downstream.core.cache import checkpoint_content_hash


def _digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_split(raw: dict) -> SubjectSplit:
    return SubjectSplit(*(frozenset(str(x) for x in raw.get(role, ())) for role in ("train", "validation", "test")))


def run_downstream(config_path: str | Path):
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    encoder_cfg = load_config(raw["encoder_config"])
    if encoder_cfg.method != "jepa":
        raise ValueError("the current formal downstream CLI supports JEPA checkpoints")
    data = raw["data"]
    legacy = ProcessedPPGDataset(data["manifest"], data["processed_root"],
                                 data.get("expected_preprocessing_version"), data.get("input_length", 5000))
    resolver_map = data.get("subject_resolver")
    subject_resolver = (lambda caseid: resolver_map[caseid]) if resolver_map else None
    dataset = ProcessedPPGUnifiedAdapter(
        legacy, sampling_rate_hz=float(data.get("sampling_rate_hz", 500.0)),
        subject_resolver=subject_resolver,
        subject_namespace=data.get("subject_namespace"),
    )
    split = _load_split(raw["split"])
    checkpoint = Path(raw["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != "generic_v1":
        raise ValueError("new downstream path requires a generic_v1 checkpoint; legacy formal checkpoints stay on compatibility paths")
    model = JEPATrainingAdapter(build_model(encoder_cfg))
    model.load_state_dict(payload["model"])
    model.freeze_encoder()
    reader = JEPARepresentationReader()
    downstream = raw["downstream"]
    representations = tuple(downstream.get("representations", ("final_layer",)))
    features = {}
    cache_manifest = []
    checkpoint_hash = checkpoint_content_hash(checkpoint)
    dataset_hash = _digest(data["manifest"])
    preprocessing_hash = hashlib.sha256(str(data.get("expected_preprocessing_version", "unknown")).encode()).hexdigest()
    for role in ("train", "validation", "test"):
        context = SplitContext.from_split(split, role)
        view = [dataset[i] for i in range(len(dataset)) if dataset[i].subject_id in context.subject_ids]
        if not view:
            raise ValueError(f"split {role!r} has no samples")
        loader = [collate_feature_samples(view)]
        extracted = extract_features(model, loader, reader=reader, context=context,
                                     representations=representations, pooling=downstream.get("pooling", "mean"),
                                     checkpoint_reference=str(checkpoint.resolve()))
        features[role] = extracted
        for name, feature in extracted.items():
            key = FeatureCacheKey(dataset_hash, hashlib.sha256(json.dumps(feature.samples, sort_keys=True).encode()).hexdigest(),
                                  preprocessing_hash, checkpoint_hash, feature.reader_name, feature.reader_version,
                                  name, feature.pooling, "1")
            cache_path = Path(downstream.get("results_root", "results")) / downstream["experiment_name"] / "feature_cache" / f"{role}__{name}.npz"
            save_feature_cache(feature, cache_path, key)
            cache_manifest.append({"split": role, "representation": name, "path": str(cache_path.resolve()), "cache_key": key.to_dict()})
    optimizers = OptimizerFactory()
    optimizers.register("sgd", torch.optim.SGD)
    protocol = EvaluationProtocol(downstream.get("task", "classification"), "linear_probe",
                                   tuple(downstream.get("metrics", ["accuracy"])), downstream.get("aggregation", "sample"), True)
    policy = DownstreamPolicy(reader.name, reader.version, downstream.get("pooling", "mean"),
                              checkpoint_selection=downstream.get("checkpoint_selection", "last"),
                              selection_metric=downstream.get("selection_metric"), selection_mode=downstream.get("selection_mode"))
    result = run_layer_probe(features, experiment_name=downstream["experiment_name"], target=downstream["target"],
                             protocol=protocol, policy=policy, optimizer_factory=optimizers,
                             epochs=int(downstream.get("epochs", 1)), batch_size=int(downstream.get("batch_size", 32)),
                             seed=int(downstream.get("seed", 0)), dataset_manifest_ref=str(Path(data["manifest"]).resolve()),
                             subject_split_ref="config://" + str(Path(config_path).resolve()), results_root=downstream.get("results_root", "results"),
                             resolved_config=raw, overwrite_run=True)
    run_path = Path(downstream.get("results_root", "results")) / downstream["experiment_name"]
    (run_path / "encoder_checkpoint.json").write_text(json.dumps({"path": str(checkpoint.resolve()), "sha256": checkpoint_hash}, indent=2) + "\n")
    (run_path / "feature_cache_manifest.json").write_text(json.dumps({"schema_version": 1, "entries": cache_manifest}, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_downstream(args.config)
    print(json.dumps({"schema_version": result["schema_version"], "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
