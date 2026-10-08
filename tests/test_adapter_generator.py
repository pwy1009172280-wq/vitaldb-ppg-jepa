"""Dedicated mask RNG ownership: adapter forwards a named generator to JEPA."""

import torch

from src.models.common import BackboneConfig
from src.models.jepa import JEPA1D, JEPATrainingAdapter


def _make(seed=None):
    torch.manual_seed(7)  # deterministic model init so loss is comparable
    model = JEPA1D(
        BackboneConfig(embed_dim=8, depth=2, num_heads=2, mlp_ratio=2.0, dropout=0.0),
        patch_size=4, patch_stride=4, num_target_blocks=1, target_block_length=1,
        predictor_dim=8, predictor_depth=1, predictor_num_heads=2,
    )
    adapter = JEPATrainingAdapter(model)
    if seed is not None:
        adapter.set_mask_generator(torch.Generator().manual_seed(seed))
    return adapter


def test_dedicated_generator_reproduces_masks_and_loss():
    wave = {"waveform": torch.randn(2, 1, 16)}
    a = _make(123)
    b = _make(123)
    out_a, out_b = a(wave), b(wave)
    assert torch.equal(out_a.context_mask, out_b.context_mask)
    assert torch.equal(out_a.target_mask, out_b.target_mask)
    assert torch.equal(out_a.loss, out_b.loss)


def test_different_generator_seed_changes_masks():
    wave = {"waveform": torch.randn(2, 1, 16)}
    a = _make(123)
    b = _make(456)
    out_a, out_b = a(wave), b(wave)
    assert not torch.equal(out_a.target_mask, out_b.target_mask)


def test_no_generator_still_forward_works():
    wave = {"waveform": torch.randn(2, 1, 16)}
    adapter = _make(None)
    out = adapter(wave)
    assert torch.isfinite(out.loss)
