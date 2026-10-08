#!/usr/bin/env python3
"""Authoritative new-experiment entry point: adapter -> Trainer -> generic checkpoint."""

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ProcessedPPGUnifiedAdapter
from src.data.processed_dataset import ProcessedPPGDataset
from src.experiments.run import create_run
from src.models.factory import build_model
from src.models.jepa import JEPATrainingAdapter
from src.pretrain.config import load_config
from src.training import CheckpointManager, CheckpointSelectionPolicy, Trainer, TrainerConfig


def _collate(samples):
    return {"waveform": torch.stack([torch.from_numpy(s.signal) for s in samples]), "samples": tuple(samples)}


def run_training(config_path: str | Path, *, run_dir: str | Path | None = None,
                 device: str | None = None, max_updates: int | None = None):
    cfg = load_config(config_path)
    if cfg.method != "jepa":
        raise ValueError("the authoritative unified entry point currently supports JEPA only")
    data, train = cfg.data, cfg.train
    target_device = torch.device(device or ("cuda" if train.device == "auto" and torch.cuda.is_available() else train.device))
    legacy = ProcessedPPGDataset(data.manifest, data.processed_root, data.expected_preprocessing_version, data.input_length)
    dataset = ProcessedPPGUnifiedAdapter(legacy)
    if not len(dataset):
        raise ValueError("processed dataset is empty")
    batches = list(DataLoader(dataset, batch_size=train.batch_size, shuffle=False,
                              drop_last=train.drop_last, num_workers=train.num_workers,
                              collate_fn=_collate))
    if not batches:
        raise ValueError("batch_size and drop_last produce zero batches")
    if max_updates is None:
        max_updates = train.max_updates
    if max_updates is not None:
        batches = batches[:max_updates]
    if not batches:
        raise ValueError("max_updates must be positive")
    model = JEPATrainingAdapter(build_model(cfg)).to(target_device)
    root = Path(run_dir or train.run_dir)
    resolved = {"method": cfg.method, "model": dict(cfg.model.__dict__), "data": dict(data.__dict__),
                "train": dict(train.__dict__), "jepa": dict(cfg.jepa.__dict__), "path": "generic"}
    run, _ = create_run(root.name, resolved, str(Path(data.manifest).resolve()),
                        "split://pretraining/all-subjects", train.seed, results_root=root.parent,
                        preprocessing_version=data.expected_preprocessing_version)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.learning_rate, weight_decay=train.weight_decay)
    manager = CheckpointManager(run.checkpoints_path, CheckpointSelectionPolicy("last"))
    trainer = Trainer(model, optimizer, config=TrainerConfig(amp=train.amp, grad_clip_norm=train.grad_clip_norm,
        scheduler_step_policy="none"), checkpoint_manager=manager,
        protocol_reference="protocol://pretraining/jepa-generic",
        experiment_manifest_reference=str(run.manifest_path.resolve()), resolved_config=resolved)
    trainer.fit(batches, epochs=train.epochs)
    return run.path / "checkpoints" / "last.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--device")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    print(run_training(args.config, run_dir=args.run_dir, device=args.device, max_updates=args.max_updates))


if __name__ == "__main__":
    main()
