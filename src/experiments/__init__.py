"""Experiment metadata and run infrastructure."""

from .manifest import ExperimentRunManifest
from .protocol import EvaluationProtocol
from .registry import ExperimentInterface, ExperimentRegistry
from .run import RunDirectory, create_run, detect_code_version, detect_git_commit, detect_git_state
from .validation import validate_pretraining_evaluation_subjects, validate_train_validation_subjects

__all__ = [
    "ExperimentInterface",
    "ExperimentRegistry",
    "ExperimentRunManifest",
    "EvaluationProtocol",
    "RunDirectory",
    "create_run",
    "detect_code_version",
    "detect_git_commit",
    "detect_git_state",
    "validate_pretraining_evaluation_subjects",
    "validate_train_validation_subjects",
]
