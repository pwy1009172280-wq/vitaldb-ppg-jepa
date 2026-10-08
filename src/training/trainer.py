"""Small, model-agnostic training execution loop."""

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .checkpoint import CheckpointManager
from .contracts import LossOutput
from .metrics import MetricsLogger


@dataclass(frozen=True)
class TrainerConfig:
    amp: bool = False
    grad_accum_steps: int = 1
    grad_clip_norm: float | None = None
    scheduler_step_policy: str = "epoch"
    max_updates: int | None = None
    max_batches: int | None = None

    def __post_init__(self) -> None:
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if self.scheduler_step_policy not in ("step", "epoch", "none"):
            raise ValueError("scheduler_step_policy must be step, epoch, or none")
        if self.max_updates is not None and self.max_updates <= 0:
            raise ValueError("max_updates must be positive")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive")


class Trainer:
    def __init__(
        self,
        model: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        *,
        config: TrainerConfig | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        metrics_logger: MetricsLogger | None = None,
        protocol_reference: str | None = None,
        experiment_manifest_reference: str | None = None,
        resolved_config: Any = None,
    ) -> None:
        if not protocol_reference:
            raise ValueError("protocol_reference is required")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainerConfig()
        self.checkpoint_manager = checkpoint_manager
        if metrics_logger is not None and metrics_logger.protocol_reference != protocol_reference:
            raise ValueError("metrics logger protocol does not match experiment protocol")
        self.metrics_logger = metrics_logger or MetricsLogger(protocol_reference=protocol_reference)
        self.experiment_manifest_reference = experiment_manifest_reference
        self.protocol_reference = protocol_reference
        self.resolved_config = resolved_config
        self.global_step = 0
        self.epoch = 0
        self.best_metric_name: str | None = None
        self.best_metric_value: float | None = None
        # Device ownership: AMP/autocast follow the model's actual device, not
        # merely whether the machine has CUDA.
        device = self._model_device()
        enabled = self.config.amp and device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _autocast(self):
        device_type = self._model_device().type
        dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=self.scaler.is_enabled())

    def _step_optimizer(self, accumulation_count: int) -> bool:
        scale_before = self.scaler.get_scale()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(accumulation_count)
        if self.config.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        did_update = self.scaler.get_scale() >= scale_before
        if did_update:
            self.model.on_optimizer_step()
            self.global_step += 1
            if self.scheduler is not None and self.config.scheduler_step_policy == "step":
                self.scheduler.step()
        return did_update

    def _run_phase(
        self,
        loader: Iterable[Any],
        epoch: int,
        training: bool,
        budget: int | None = None,
        attempts_cap: int | None = None,
    ) -> dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        batches = 0
        pending = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch_index, batch in enumerate(loader):
                with self._autocast() if training else torch.no_grad():
                    output = self.model.forward(batch)
                    loss_output = self.model.compute_loss(output, batch)
                    if not isinstance(loss_output, LossOutput):
                        raise TypeError("compute_loss must return LossOutput")
                    terms = loss_output.detached_terms()
                    loss = loss_output.total_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch} batch {batch_index}"
                    )
                for name, value in terms.items():
                    sums[name] = sums.get(name, 0.0) + value
                batches += 1
                if training:
                    self.scaler.scale(loss).backward()
                    pending += 1
                    is_last = batch_index == len(loader) - 1 if hasattr(loader, "__len__") else False
                    if pending == self.config.grad_accum_steps or is_last:
                        self._step_optimizer(pending)
                        self.optimizer.zero_grad(set_to_none=True)
                        pending = 0
                if budget is not None and self.global_step >= budget:
                    break
                if attempts_cap is not None and batches >= attempts_cap:
                    break
            if training and pending:
                self._step_optimizer(pending)
                self.optimizer.zero_grad(set_to_none=True)
        if batches == 0:
            raise ValueError(f"empty {('train' if training else 'validation')} loader at epoch {epoch}")
        averages = {name: value / max(1, batches) for name, value in sums.items()}
        self.metrics_logger.log("train" if training else "validation", epoch, self.global_step, averages, batches)
        return averages

    def fit(
        self,
        train_loader: Iterable[Any],
        *,
        validation_loader: Iterable[Any] | None = None,
        epochs: int = 1,
        max_updates: int | None = None,
        max_batches: int | None = None,
    ) -> list[dict[str, float]]:
        if self.checkpoint_manager is not None and self.checkpoint_manager.policy.kind == "monitored_metric" and validation_loader is None:
            raise ValueError("monitored_metric checkpoint selection requires validation_loader")
        budget = max_updates if max_updates is not None else self.config.max_updates
        attempts_cap = max_batches if max_batches is not None else self.config.max_batches
        history: list[dict[str, float]] = []
        for epoch in range(self.epoch, self.epoch + epochs):
            self.epoch = epoch
            steps_before_epoch = self.global_step
            train_metrics = self._run_phase(train_loader, epoch, True, budget=budget, attempts_cap=attempts_cap)
            record = {f"train/{name}": value for name, value in train_metrics.items()}
            if validation_loader is not None:
                validation_metrics = self._run_phase(validation_loader, epoch, False)
                record.update({f"validation/{name}": value for name, value in validation_metrics.items()})
            history.append(record)
            if (
                self.scheduler is not None
                and self.config.scheduler_step_policy == "epoch"
                and self.global_step > steps_before_epoch
            ):
                self.scheduler.step()
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save(
                    "last", self.model, self.optimizer, self.scheduler, self.scaler,
                    epoch=epoch, global_step=self.global_step,
                    best_metric_name=self.best_metric_name, best_metric_value=self.best_metric_value,
                    experiment_manifest_reference=self.experiment_manifest_reference,
                    evaluation_protocol_reference=self.protocol_reference,
                    resolved_config=self.resolved_config,
                )
                if validation_loader is not None and self.checkpoint_manager.policy.kind == "monitored_metric":
                    monitor = self.checkpoint_manager.policy.monitor
                    candidate = validation_metrics.get(monitor) if monitor else None
                    if candidate is None:
                        raise ValueError(f"monitored metric {monitor!r} was not produced by validation")
                    better = self.best_metric_value is None or (
                        candidate < self.best_metric_value
                        if self.checkpoint_manager.policy.mode == "min"
                        else candidate > self.best_metric_value
                    )
                    if better:
                        self.best_metric_name = monitor
                        self.best_metric_value = candidate
                        self.checkpoint_manager.save_best(
                            self.model, self.optimizer, self.scheduler, self.scaler,
                            epoch=epoch, global_step=self.global_step,
                            best_metric_name=self.best_metric_name, best_metric_value=self.best_metric_value,
                            experiment_manifest_reference=self.experiment_manifest_reference,
                            evaluation_protocol_reference=self.protocol_reference,
                            resolved_config=self.resolved_config,
                        )
                        self.checkpoint_manager.save(
                            "last", self.model, self.optimizer, self.scheduler, self.scaler,
                            epoch=epoch, global_step=self.global_step,
                            best_metric_name=self.best_metric_name, best_metric_value=self.best_metric_value,
                            experiment_manifest_reference=self.experiment_manifest_reference,
                            evaluation_protocol_reference=self.protocol_reference,
                            resolved_config=self.resolved_config,
                        )
            self.epoch = epoch + 1
            if budget is not None and self.global_step >= budget:
                break
        return history

    def resume(self, selection: str = "last") -> dict[str, Any]:
        if self.checkpoint_manager is None:
            raise RuntimeError("checkpoint_manager is required for resume")
        payload = self.checkpoint_manager.load(
            selection, self.model, self.optimizer, self.scheduler, self.scaler,
            expected_resolved_config=self.resolved_config,
            expected_experiment_manifest_reference=self.experiment_manifest_reference,
            expected_evaluation_protocol_reference=self.protocol_reference,
        )
        self.epoch = int(payload["epoch"]) + 1
        self.global_step = int(payload["global_step"])
        self.best_metric_name = payload["best_metric_name"]
        self.best_metric_value = payload["best_metric_value"]
        return payload
