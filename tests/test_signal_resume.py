"""Signal-safe intra-epoch resume (A8 durability)."""

import threading

import pytest
import torch

from src.training import CheckpointManager, CheckpointSelectionPolicy, Trainer, TrainerConfig
from src.training.contracts import LossOutput
from src.training.trainer import TrainingInterrupted


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return self.weight * batch["x"]

    def compute_loss(self, output, batch):
        loss = (output - batch["y"]).square().mean()
        return LossOutput(loss, {"objective": loss})

    def on_optimizer_step(self):
        pass


def _batch():
    return {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])}


class StoppingLoader:
    def __init__(self, n, stop_after, event):
        self.n = n
        self.stop_after = stop_after
        self.event = event

    def __iter__(self):
        for i in range(self.n):
            yield _batch()
            if i + 1 == self.stop_after:
                self.event.set()

    def __len__(self):
        return self.n


def _trainer(model, manager):
    return Trainer(
        model, torch.optim.SGD(model.parameters(), lr=1.0),
        config=TrainerConfig(scheduler_step_policy="none"),
        checkpoint_manager=manager,
        protocol_reference="protocol://toy",
        experiment_manifest_reference="manifest://toy",
        resolved_config={"seed": 1},
    )


def test_signal_safe_intra_epoch_resume(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointSelectionPolicy("last"))
    trainer = _trainer(Toy(), manager)

    stop = threading.Event()
    with pytest.raises(TrainingInterrupted) as exc:
        trainer.fit(StoppingLoader(5, 3, stop), epochs=1, stop_event=stop)
    interrupted_steps = trainer.global_step
    assert interrupted_steps < 5  # did not finish the epoch

    resumed = _trainer(Toy(), manager)
    payload = resumed.resume("last")
    assert payload["epoch_complete"] is False
    assert payload["next_batch_idx"] == exc.value.next_batch_idx
    assert resumed.epoch == 0
    assert resumed._next_batch_idx == exc.value.next_batch_idx

    resumed.fit(StoppingLoader(5, 99, threading.Event()), epochs=1)
    # interrupted + resumed total equals the uninterrupted 5-batch trace
    assert resumed.global_step == 5


def test_epoch_complete_resume_advances_epoch(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointSelectionPolicy("last"))
    trainer = _trainer(Toy(), manager)
    trainer.fit(StoppingLoader(2, 99, threading.Event()), epochs=1)
    payload = torch.load(manager._path("last"), weights_only=False)
    assert payload["epoch_complete"] is True
    resumed = _trainer(Toy(), manager)
    resumed.resume("last")
    assert resumed.epoch == 1
    assert resumed._next_batch_idx == 0
