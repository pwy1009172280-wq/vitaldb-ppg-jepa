import json

import pytest
import torch

from src.training import (
    CheckpointManager,
    CheckpointSelectionPolicy,
    LossOutput,
    MetricsLogger,
    OptimizerFactory,
    SchedulerFactory,
    Trainer,
    TrainerConfig,
)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.optimizer_steps = 0

    def forward(self, batch):
        return self.weight * batch["x"]

    def compute_loss(self, output, batch):
        prediction_loss = ((output - batch["y"]) ** 2).mean()
        regularization = self.weight.square() * 0.01
        return LossOutput(prediction_loss + regularization, {
            "predictor_loss": prediction_loss,
            "regularization_loss": regularization,
        })

    def on_optimizer_step(self):
        self.optimizer_steps += 1


def batches():
    return [
        {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])},
        {"x": torch.tensor([2.0]), "y": torch.tensor([4.0])},
    ]


def make_optimizer(model):
    factory = OptimizerFactory()
    factory.register("sgd", torch.optim.SGD)
    return factory.create("sgd", model.parameters(), lr=0.1)


def test_toy_model_training_lifecycle_and_structured_logging(tmp_path):
    model = ToyModel()
    optimizer = make_optimizer(model)
    scheduler_factory = SchedulerFactory()
    scheduler_factory.register("step", torch.optim.lr_scheduler.StepLR)
    scheduler = scheduler_factory.create("step", optimizer, step_size=1, gamma=0.9)
    metrics_path = tmp_path / "metrics.json"
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        config=TrainerConfig(grad_accum_steps=2, scheduler_step_policy="epoch"),
        metrics_logger=MetricsLogger(metrics_path, protocol_reference="protocol://toy"),
        protocol_reference="protocol://toy",
    )
    history = trainer.fit(batches(), epochs=2)
    assert len(history) == 2
    assert model.optimizer_steps == 2
    assert model.training
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert records[0]["loss_terms"]["predictor_loss"] >= 0
    assert "regularization_loss" in records[0]["loss_terms"]
    assert records[0]["protocol_reference"] == "protocol://toy"
    assert records[0]["protocol_hash"]


def test_checkpoint_save_load_and_resume_restores_state(tmp_path):
    model = ToyModel()
    optimizer = make_optimizer(model)
    manager = CheckpointManager(tmp_path, CheckpointSelectionPolicy("last"))
    trainer = Trainer(model, optimizer, checkpoint_manager=manager, protocol_reference="protocol://toy")
    trainer.fit(batches(), epochs=1)
    saved_weight = model.weight.detach().clone()
    saved_step = trainer.global_step
    with torch.no_grad():
        model.weight.add_(100)

    resumed_model = ToyModel()
    resumed_optimizer = make_optimizer(resumed_model)
    resumed = Trainer(resumed_model, resumed_optimizer, checkpoint_manager=manager, protocol_reference="protocol://toy")
    payload = resumed.resume("last")
    assert resumed.global_step == saved_step
    assert resumed.epoch == 1
    assert torch.equal(resumed_model.weight, saved_weight)
    assert payload["resolved_config"] is None


def test_monitored_metric_best_policy_is_explicit(tmp_path):
    model = ToyModel()
    manager = CheckpointManager(
        tmp_path,
        CheckpointSelectionPolicy(kind="monitored_metric", monitor="total_loss", mode="min"),
    )
    trainer = Trainer(model, make_optimizer(model), checkpoint_manager=manager, protocol_reference="protocol://toy")
    trainer.fit(batches(), validation_loader=batches(), epochs=1)
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert not model.training
    best_model = ToyModel()
    assert manager.load(
        "best", best_model, make_optimizer(best_model), None, None,
        expected_resolved_config=None,
        expected_experiment_manifest_reference=None,
        expected_evaluation_protocol_reference="protocol://toy",
    )["best_metric_name"] == "total_loss"


def test_amp_switch_and_multi_view_opaque_batch():
    model = ToyModel()
    trainer = Trainer(model, make_optimizer(model), config=TrainerConfig(amp=True), protocol_reference="protocol://toy")
    multi_view_batch = {"views": [torch.tensor([1.0]), torch.tensor([2.0])], "x": torch.tensor([1.0]), "y": torch.tensor([2.0])}
    trainer.fit([multi_view_batch], epochs=1)
    assert trainer.config.amp is True


def test_invalid_checkpoint_policy_is_rejected():
    with pytest.raises(ValueError):
        CheckpointSelectionPolicy(kind="monitored_metric", monitor="loss", mode="median")
