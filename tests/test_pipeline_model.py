"""Model construction and reconstruction from the generic config (A5/A8 glue)."""

import pytest
import torch

from src.experiments.pipeline import build_model_from_config, reconstruct_model, resolve_device


TINY_MODEL = {
    "family": "jepa",
    "patch_size": 4, "patch_stride": 4, "embed_dim": 8, "depth": 2,
    "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0,
    "decoder_dim": 4, "decoder_depth": 1, "decoder_num_heads": 2, "decoder_mlp_ratio": 2.0,
    "num_target_blocks": 1, "target_block_length": 1, "predictor_dim": 8,
    "predictor_depth": 1, "predictor_num_heads": 2, "predictor_mlp_ratio": 2.0,
    "ema_momentum": 0.9, "loss_beta": 1.0,
}


def test_build_model_from_config_constructs_jepa():
    model = build_model_from_config(TINY_MODEL)
    assert hasattr(model, "encode_full")
    assert hasattr(model, "ema_update")


def test_build_model_rejects_non_jepa_family():
    with pytest.raises(ValueError, match="family"):
        build_model_from_config({**TINY_MODEL, "family": "mae"})


def test_reconstruct_model_from_resolved_config():
    model = reconstruct_model({"model": TINY_MODEL})
    assert isinstance(model, torch.nn.Module)
    with pytest.raises(ValueError, match="model"):
        reconstruct_model({"dataset": {}})


def test_resolve_device():
    assert isinstance(resolve_device("cpu"), torch.device)
    assert resolve_device("cpu").type == "cpu"
    assert isinstance(resolve_device(None), torch.device)
