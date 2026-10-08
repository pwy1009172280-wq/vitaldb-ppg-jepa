"""Parity-gate tests for the generic Trainer: fail-fast and true update budget."""

import pytest
import torch

from src.training import Trainer, TrainerConfig
from src.training.contracts import LossOutput


class Toy(torch.nn.Module):
    def __init__(self, *, nan: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.nan = nan
        self.steps = 0

    def forward(self, batch):
        return self.weight * batch["x"]

    def compute_loss(self, output, batch):
        loss = (output - batch["y"]).square().mean()
        if self.nan:
            loss = loss * float("nan")
        return LossOutput(loss, {"objective": loss})

    def on_optimizer_step(self):
        self.steps += 1


def _trainer(model, *, config=None):
    return Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=1.0),
        config=config,
        protocol_reference="protocol://toy",
    )


def _batch():
    return {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])}


def test_nonfinite_loss_fails_fast():
    model = Toy(nan=True)
    trainer = _trainer(model)
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.fit([_batch()], epochs=1)


def test_empty_loader_fails():
    model = Toy()
    trainer = _trainer(model)
    with pytest.raises(ValueError, match="empty"):
        trainer.fit([], epochs=1)


def test_max_updates_budget_counts_successful_updates_across_epochs():
    model = Toy()
    trainer = _trainer(model, config=TrainerConfig(scheduler_step_policy="none"))
    trainer.fit([_batch(), _batch(), _batch(), _batch()], epochs=5, max_updates=2)
    assert trainer.global_step == 2
    assert model.steps == 2


def test_max_batches_caps_attempts():
    model = Toy()
    trainer = _trainer(model, config=TrainerConfig(scheduler_step_policy="none", grad_accum_steps=2))
    # 4 batches with accum=2 would be 2 updates; cap attempts at 2 batches -> 1 update
    trainer.fit([_batch(), _batch(), _batch(), _batch()], epochs=1, max_batches=2)
    assert trainer.global_step == 1
    assert model.steps == 1


def test_trainer_config_validates_budget():
    with pytest.raises(ValueError, match="max_updates"):
        TrainerConfig(max_updates=0)
    with pytest.raises(ValueError, match="max_batches"):
        TrainerConfig(max_batches=-1)
