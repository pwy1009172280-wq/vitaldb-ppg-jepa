"""Checkpoint self-description: reconstruction, epoch_complete, external mismatch."""

import pytest
import torch

from src.experiments.pipeline import build_model_from_config, load_checkpoint_model
from src.models.jepa import JEPATrainingAdapter
from src.training import CheckpointManager, CheckpointSelectionPolicy

TINY_MODEL = {
    "family": "jepa",
    "patch_size": 4, "patch_stride": 4, "embed_dim": 8, "depth": 2,
    "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0,
    "decoder_dim": 4, "decoder_depth": 1, "decoder_num_heads": 2, "decoder_mlp_ratio": 2.0,
    "num_target_blocks": 1, "target_block_length": 1, "predictor_dim": 8,
    "predictor_depth": 1, "predictor_num_heads": 2, "predictor_mlp_ratio": 2.0,
    "ema_momentum": 0.9, "loss_beta": 1.0,
}


def _save(tmp_path, *, resolved=None):
    raw = build_model_from_config(TINY_MODEL)
    adapter = JEPATrainingAdapter(raw)
    manager = CheckpointManager(tmp_path, CheckpointSelectionPolicy("last"))
    manager.save(
        "last", adapter, torch.optim.SGD(adapter.parameters(), lr=1e-3), None, None,
        epoch=0, global_step=2, best_metric_name=None, best_metric_value=None,
        experiment_manifest_reference="manifest://x",
        evaluation_protocol_reference="protocol://x",
        resolved_config=resolved or {"model": TINY_MODEL},
        epoch_complete=True,
    )
    return raw, str(tmp_path / "last.pt")


def test_checkpoint_reconstruction_and_epoch_complete(tmp_path):
    raw, path = _save(tmp_path)
    rebuilt, payload = load_checkpoint_model(path)
    assert payload["epoch_complete"] is True
    assert payload["global_step"] == 2
    for (_, p1), (_, p2) in zip(raw.state_dict().items(), rebuilt.jepa.state_dict().items()):
        assert torch.equal(p1, p2)


def test_checkpoint_reconstruction_rejects_external_mismatch(tmp_path):
    _, path = _save(tmp_path)
    load_checkpoint_model(path, expected_model_section=TINY_MODEL)
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint_model(path, expected_model_section={**TINY_MODEL, "depth": 3})


def test_checkpoint_reconstruction_rejects_legacy_format(tmp_path):
    p = tmp_path / "legacy.pt"
    torch.save({"checkpoint_format": "other", "model": {}, "resolved_config": {}}, p)
    with pytest.raises(ValueError, match="generic_v1"):
        load_checkpoint_model(str(p))


def test_checkpoint_reconstruction_requires_model_section(tmp_path):
    _, path = _save(tmp_path, resolved={"dataset": {}})
    with pytest.raises(ValueError, match="model"):
        load_checkpoint_model(path)
