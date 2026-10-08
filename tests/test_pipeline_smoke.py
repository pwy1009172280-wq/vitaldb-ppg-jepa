"""Fixture end-to-end smoke (A14): pretrain -> representation -> cache -> result."""

import numpy as np

from src.config import PipelineConfig
from src.data.base import BaseDataset
from src.data.samples import SUBJECT_IDENTITY_RESOLVED, UnifiedSample
from src.experiments.pipeline import run_smoke

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
        if index < 0 or index >= self.n:
            raise IndexError(index)
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


def _config():
    return PipelineConfig(
        model=TINY_MODEL,
        experiment={
            "mode": "smoke", "smoke_only": True, "seed": 123, "device": "cpu",
            "training": {"batch_size": 2, "max_updates": 2, "lr": 1e-3, "weight_decay": 0.0},
        },
    )


def test_fixture_smoke_end_to_end(tmp_path):
    out = run_smoke(_config(), SyntheticDataset(), results_root=tmp_path / "results", run_id="smoke-run")
    result = out["result"]
    assert result["metrics"]["accuracy"] == 1.0
    assert set(result["representations"]) == {"layer_1", "layer_2", "final_layer"}
    assert result["smoke_only"] is True
    assert (tmp_path / "results" / "smoke-run" / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "results" / "smoke-run" / "metrics.json").exists()
    assert (tmp_path / "results" / "smoke-run" / "feature_cache" / "final_layer.npz").exists()
