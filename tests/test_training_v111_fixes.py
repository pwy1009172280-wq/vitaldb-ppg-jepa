import json

import pytest
import torch

from src.experiments import validate_pretraining_evaluation_subjects, validate_train_validation_subjects
from src.training import (
    CheckpointManager,
    CheckpointSelectionPolicy,
    LossOutput,
    MetricsLogger,
    Trainer,
    TrainerConfig,
)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.hook_calls = 0

    def forward(self, batch):
        return self.weight * batch["x"]

    def compute_loss(self, output, batch):
        loss = (output - batch["y"]).square().mean()
        return LossOutput(loss, {"objective": loss})

    def on_optimizer_step(self):
        self.hook_calls += 1


def batch():
    return {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])}


def make_trainer(model=None, manager=None, *, config=None, protocol="protocol://toy", resolved=None, manifest="manifest://toy"):
    model = model or Toy()
    return Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=1.0),
        config=config,
        checkpoint_manager=manager,
        protocol_reference=protocol,
        experiment_manifest_reference=manifest,
        resolved_config=resolved,
    )


class SkippingScaler:
    def __init__(self):
        self.scale = 2.0

    def get_scale(self):
        return self.scale

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        pass

    def update(self):
        self.scale = 1.0

    def state_dict(self):
        return {}


def test_amp_skipped_step_does_not_advance_lifecycle():
    model = Toy()
    trainer = make_trainer(model)
    trainer.scaler = SkippingScaler()
    assert trainer._step_optimizer(1) is False
    assert model.hook_calls == 0
    assert trainer.global_step == 0


def test_accumulation_tail_is_averaged_by_actual_group_size():
    one_step = make_trainer()
    one_step.fit([batch()], epochs=1)
    accumulated = make_trainer(config=TrainerConfig(grad_accum_steps=4))
    accumulated.fit([batch()], epochs=1)
    assert accumulated.model.weight.item() == pytest.approx(one_step.model.weight.item())
    assert accumulated.model.hook_calls == 1


def test_resume_identity_mismatch_fails(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointSelectionPolicy("last"))
    first = make_trainer(manager=manager, resolved={"seed": 1})
    first.fit([batch()], epochs=1)
    mismatch = make_trainer(manager=manager, resolved={"seed": 2})
    with pytest.raises(ValueError, match="resolved config"):
        mismatch.resume()
    mismatch_manifest = make_trainer(manager=manager, resolved={"seed": 1}, manifest="manifest://other")
    with pytest.raises(ValueError, match="manifest"):
        mismatch_manifest.resume()


def test_resume_protocol_and_policy_mismatch_fail(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointSelectionPolicy("last"))
    first = make_trainer(manager=manager)
    first.fit([batch()], epochs=1)
    with pytest.raises(ValueError, match="protocol"):
        make_trainer(manager=manager, protocol="protocol://other").resume()
    other_policy = CheckpointManager(tmp_path, CheckpointSelectionPolicy("monitored_metric", "objective", "min"))
    with pytest.raises(ValueError, match="policy"):
        make_trainer(manager=other_policy).resume()


def test_metrics_resume_appends_history(tmp_path):
    path = tmp_path / "metrics.json"
    first = MetricsLogger(path, "protocol://toy")
    first.log("train", 0, 1, {"objective": 2.0}, 1)
    resumed = MetricsLogger(path, "protocol://toy")
    resumed.log("train", 1, 2, {"objective": 1.0}, 1)
    records = json.loads(path.read_text(encoding="utf-8"))
    assert [record["epoch"] for record in records] == [0, 1]


def test_split_overlap_validation_fails():
    with pytest.raises(ValueError):
        validate_train_validation_subjects({"s1"}, {"s1"})
    with pytest.raises(ValueError):
        validate_pretraining_evaluation_subjects({"s1"}, {"s1"})


def test_missing_monitored_metric_fails(tmp_path):
    manager = CheckpointManager(
        tmp_path, CheckpointSelectionPolicy("monitored_metric", "missing", "min")
    )
    trainer = make_trainer(manager=manager)
    with pytest.raises(ValueError, match="monitored metric"):
        trainer.fit([batch()], validation_loader=[batch()], epochs=1)


def test_epoch_cursor_resumes_continuously(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointSelectionPolicy("last"))
    first = make_trainer(manager=manager)
    first.fit([batch()], epochs=1)
    assert first.epoch == 1
    resumed = make_trainer(manager=manager)
    resumed.resume()
    resumed.fit([batch()], epochs=1)
    assert resumed.epoch == 2
    assert [record["epoch"] for record in resumed.metrics_logger.records] == [1]


def test_loss_terms_cannot_shadow_total_loss():
    with pytest.raises(ValueError, match="total_loss"):
        LossOutput(torch.tensor(1.0), {"total_loss": torch.tensor(2.0)})
