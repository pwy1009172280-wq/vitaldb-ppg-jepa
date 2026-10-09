"""Tests for the formal pretraining flow (checkpoint interval + scheduler)."""

import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import PipelineConfig
from src.experiments.metadata import seed_everything
from src.experiments.pipeline import build_model_from_config
from src.models.jepa import JEPATrainingAdapter
from src.training import CheckpointManager, CheckpointSelectionPolicy, Trainer, TrainerConfig
from src.training.schedules import build_warmup_cosine_scheduler

TINY = {
    "family": "jepa",
    "patch_size": 4, "patch_stride": 4, "embed_dim": 8, "depth": 2,
    "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0,
    "decoder_dim": 4, "decoder_depth": 1, "decoder_num_heads": 2, "decoder_mlp_ratio": 2.0,
    "num_target_blocks": 1, "target_block_length": 1, "predictor_dim": 8,
    "predictor_depth": 1, "predictor_num_heads": 2, "predictor_mlp_ratio": 2.0,
    "ema_momentum": 0.9, "loss_beta": 1.0,
}


def test_formal_pretrain_checkpoint_interval(tmp_path):
    seed_everything(11)
    model = build_model_from_config(TINY)
    adapter = JEPATrainingAdapter(model)
    adapter.set_mask_generator(torch.Generator().manual_seed(11 + 7919))

    windows = torch.randn(16, 1, 16)
    collate = lambda batch: {"waveform": torch.stack([x[0] for x in batch])}  # noqa: E731
    loader = DataLoader(TensorDataset(windows), batch_size=2, shuffle=True, drop_last=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=2, total_steps=6, min_lr_ratio=0.01)

    manager = CheckpointManager(tmp_path, CheckpointSelectionPolicy("last"))
    trainer = Trainer(
        adapter, optimizer, scheduler,
        config=TrainerConfig(
            amp=False, grad_clip_norm=1.0, scheduler_step_policy="step",
            max_updates=6, checkpoint_interval_updates=2,
        ),
        checkpoint_manager=manager,
        protocol_reference="protocol://test/formal",
        resolved_config={"model": TINY},
    )
    trainer.fit(loader, epochs=3, max_updates=6)
    assert trainer.global_step == 6
    # checkpoints: step-2, step-4, step-6 (interval) + last (epoch end)
    for name in ("step-2", "step-4", "step-6", "last"):
        assert (tmp_path / f"{name}.pt").exists(), f"missing {name}"
    # scheduler stepped per update
    assert scheduler.last_epoch == 6
