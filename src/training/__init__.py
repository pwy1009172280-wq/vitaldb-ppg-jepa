"""Generic biosignal training execution layer."""

from .checkpoint import CheckpointManager, CheckpointSelectionPolicy
from .contracts import LossOutput, TrainingModel
from .metrics import MetricsLogger
from .optim import OptimizerFactory, SchedulerFactory
from .trainer import Trainer, TrainerConfig

__all__ = [
    "CheckpointManager",
    "CheckpointSelectionPolicy",
    "LossOutput",
    "MetricsLogger",
    "OptimizerFactory",
    "SchedulerFactory",
    "Trainer",
    "TrainerConfig",
    "TrainingModel",
]
