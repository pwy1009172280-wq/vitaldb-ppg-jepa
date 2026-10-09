"""Lock the experiment model's all-layer extraction contract (plan §11)."""

import torch

from src.experiments.pipeline import build_model_from_config

MODEL = {
    "family": "jepa",
    "patch_size": 25, "patch_stride": 25, "embed_dim": 128, "depth": 4,
    "num_heads": 4, "mlp_ratio": 4.0, "dropout": 0.0,
    "decoder_dim": 64, "decoder_depth": 2, "decoder_num_heads": 4, "decoder_mlp_ratio": 4.0,
    "num_target_blocks": 2, "target_block_length": 10, "predictor_dim": 128,
    "predictor_depth": 2, "predictor_num_heads": 4, "predictor_mlp_ratio": 4.0,
    "ema_momentum": 0.996, "loss_beta": 1.0,
}


def test_experiment_model_all_layer_extraction():
    model = build_model_from_config(MODEL)
    model.eval()
    x = torch.randn(2, 1, 2000)
    with torch.no_grad():
        out = model.encode_full(x)
    hs = out.hidden_states
    assert len(hs) == 4, f"expected 4 blocks, got {len(hs)}"
    for h in hs:
        assert tuple(h.shape) == (2, 80, 128)
    assert tuple(out.final_tokens.shape) == (2, 80, 128)
    # mean-token pooling -> [B, 128]
    for h in hs:
        assert tuple(h.mean(dim=1).shape) == (2, 128)
    assert tuple(out.final_tokens.mean(dim=1).shape) == (2, 128)


def test_experiment_model_param_count():
    model = build_model_from_config(MODEL)
    total = sum(p.numel() for p in model.parameters())
    assert total == 2023296  # frozen resource estimate (plan §20)
