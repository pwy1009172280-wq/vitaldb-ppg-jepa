"""Pipeline pretrain orchestration (seed-before-init, groups, mask RNG, budget)."""

import numpy as np
import torch

from src.config import PipelineConfig
from src.data.base import BaseDataset
from src.data.samples import SUBJECT_IDENTITY_RESOLVED, UnifiedSample
from src.experiments.pipeline import load_checkpoint_model, run_pretrain

TINY_MODEL = {
    "family": "jepa",
    "patch_size": 4, "patch_stride": 4, "embed_dim": 8, "depth": 2,
    "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0,
    "decoder_dim": 4, "decoder_depth": 1, "decoder_num_heads": 2, "decoder_mlp_ratio": 2.0,
    "num_target_blocks": 1, "target_block_length": 1, "predictor_dim": 8,
    "predictor_depth": 1, "predictor_num_heads": 2, "predictor_mlp_ratio": 2.0,
    "ema_momentum": 0.9, "loss_beta": 1.0,
}


class SyntheticDataset(BaseDataset):
    name = "test-fixture-ppg"

    def __init__(self, n=8, window_len=16):
        self.n = n
        self.window_len = window_len

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return UnifiedSample(
            signal=np.full((1, self.window_len), float(index), dtype=np.float32),
            subject_id=f"fixture-s{index % 2}",
            recording_id=f"r{index}",
            dataset=self.name, modality="PPG", sampling_rate_hz=125.0,
            start_time_s=0.0, end_time_s=self.window_len / 125.0,
            window_id=f"w{index}", window_start_sample=0, window_end_sample=self.window_len,
            subject_identity_status=SUBJECT_IDENTITY_RESOLVED,
            subject_identity_namespace="test-fixture",
        )


def _config(seed=123, max_updates=2):
    return PipelineConfig(
        model=TINY_MODEL,
        experiment={
            "mode": "smoke", "smoke_only": True, "seed": seed, "device": "cpu",
            "training": {"batch_size": 2, "max_updates": max_updates, "lr": 1e-3, "weight_decay": 0.0},
        },
    )


def test_run_pretrain_produces_reconstructable_checkpoint(tmp_path):
    ckpt = run_pretrain(_config(), SyntheticDataset(), results_root=tmp_path / "results", run_id="fixture-run")
    assert ckpt.exists()
    model, payload = load_checkpoint_model(str(ckpt))
    assert payload["global_step"] == 2
    assert payload["epoch_complete"] is True
    assert payload["resolved_config"]["model"]["family"] == "jepa"


def test_seed_before_init_is_reproducible(tmp_path):
    cfg = _config(seed=7, max_updates=1)
    ckpt1 = run_pretrain(cfg, SyntheticDataset(), results_root=tmp_path / "r1", run_id="run-a")
    ckpt2 = run_pretrain(cfg, SyntheticDataset(), results_root=tmp_path / "r2", run_id="run-b")
    m1, _ = load_checkpoint_model(str(ckpt1))
    m2, _ = load_checkpoint_model(str(ckpt2))
    for (_, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b)
