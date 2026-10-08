"""Synthetic end-to-end integration tests for JEPA and the generic Trainer."""

from copy import deepcopy

import pytest
import torch

from src.models.common import BackboneConfig
from src.models.jepa import JEPA1D, JEPATrainingAdapter
from src.training import (
    CheckpointManager,
    CheckpointSelectionPolicy,
    Trainer,
    TrainerConfig,
)


def make_adapter() -> JEPATrainingAdapter:
    model = JEPA1D(
        BackboneConfig(embed_dim=8, depth=2, num_heads=2, mlp_ratio=2.0, dropout=0.0),
        patch_size=4,
        patch_stride=4,
        num_target_blocks=1,
        target_block_length=1,
        predictor_dim=8,
        predictor_depth=1,
        predictor_num_heads=2,
    )
    return JEPATrainingAdapter(model)


def synthetic_batches(count: int = 3):
    generator = torch.Generator().manual_seed(123)
    return [{"waveform": torch.randn(2, 1, 16, generator=generator)} for _ in range(count)]


def make_trainer(model, manager=None, *, resolved_config=None):
    return Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        config=TrainerConfig(grad_accum_steps=2, scheduler_step_policy="none"),
        checkpoint_manager=manager,
        protocol_reference="protocol://jepa-smoke",
        experiment_manifest_reference="manifest://jepa-smoke",
        resolved_config=resolved_config or {"model": "synthetic-jepa", "seed": 123},
    )


def test_jepa_adapter_maps_existing_loss_and_calls_ema_after_updates(tmp_path):
    adapter = make_adapter()
    trainer = make_trainer(adapter)
    ema_calls = []
    original_ema_update = adapter.jepa.ema_update

    def tracked_ema_update():
        ema_calls.append(True)
        original_ema_update()

    adapter.jepa.ema_update = tracked_ema_update
    target_before = [parameter.detach().clone() for parameter in adapter.jepa.target_encoder.parameters()]

    history = trainer.fit(synthetic_batches(), epochs=1)

    assert len(history) == 1
    assert trainer.global_step == 2  # one full group and one accumulation tail
    assert len(ema_calls) == trainer.global_step
    target_after = list(adapter.jepa.target_encoder.parameters())
    assert any(not torch.equal(before, after) for before, after in zip(target_before, target_after))


class SkippingScaler:
    def __init__(self):
        self._scale = 2.0

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        pass

    def update(self):
        self._scale = 1.0

    def state_dict(self):
        return {}


def test_jepa_adapter_ema_hook_is_not_called_for_amp_skipped_step():
    adapter = make_adapter()
    trainer = make_trainer(adapter)
    trainer.scaler = SkippingScaler()
    ema_calls = []
    adapter.jepa.ema_update = lambda: ema_calls.append(True)
    parameter = next(adapter.parameters())
    parameter.grad = torch.zeros_like(parameter)

    assert trainer._step_optimizer(1) is False
    assert ema_calls == []
    assert trainer.global_step == 0


def test_generic_checkpoint_reload_freeze_and_representation_smoke(tmp_path):
    resolved_config = {"model": "synthetic-jepa", "seed": 123}
    manager = CheckpointManager(tmp_path / "checkpoints", CheckpointSelectionPolicy("last"))
    adapter = make_adapter()
    trainer = make_trainer(adapter, manager, resolved_config=resolved_config)
    trainer.fit(synthetic_batches(), epochs=1)

    saved_state = {name: value.detach().clone() for name, value in adapter.state_dict().items()}
    checkpoint = tmp_path / "checkpoints" / "last.pt"
    assert checkpoint.exists()

    reloaded = make_adapter()
    resumed = make_trainer(reloaded, manager, resolved_config=resolved_config)
    payload = resumed.resume("last")

    assert payload["model"]
    assert all(torch.equal(saved_state[name], value) for name, value in reloaded.state_dict().items())
    assert all(
        torch.equal(source, target)
        for source, target in zip(
            adapter.jepa.target_encoder.parameters(), reloaded.jepa.target_encoder.parameters()
        )
    )
    output = reloaded({"waveform": torch.randn(2, 1, 16, generator=torch.Generator().manual_seed(9))})
    assert torch.isfinite(output.loss)

    reloaded.freeze_encoder()
    waveform = torch.randn(2, 1, 16, generator=torch.Generator().manual_seed(11))
    representation = reloaded.encode_full(waveform)

    assert len(representation.hidden_states) == 2
    assert all(state.shape == (2, 4, 8) for state in representation.hidden_states)
    assert representation.final_tokens.shape == (2, 4, 8)
    assert torch.isfinite(representation.final_tokens).all()
    assert all(torch.isfinite(state).all() for state in representation.hidden_states)
    assert all(not parameter.requires_grad for parameter in reloaded.encoder_parameters())
    assert all(parameter.grad is None for parameter in reloaded.encoder_parameters())


def test_adapter_rejects_non_waveform_batches():
    adapter = make_adapter()
    with pytest.raises(TypeError, match="waveform"):
        adapter({"signal": torch.randn(1, 1, 16)})
