"""Validation performed at experiment assembly time, outside Trainer."""

from collections.abc import Iterable

from src.data.leakage import assert_pretraining_downstream_disjoint, assert_subject_sets_disjoint


def validate_train_validation_subjects(
    train_subjects: Iterable[str], validation_subjects: Iterable[str]
) -> None:
    """Reject overlapping train and validation subject sets before execution."""
    assert_subject_sets_disjoint(train_subjects, validation_subjects)


def validate_pretraining_evaluation_subjects(
    pretraining_subjects: Iterable[str], evaluation_subjects: Iterable[str]
) -> None:
    """Reject downstream evaluation subjects seen during pretraining."""
    assert_pretraining_downstream_disjoint(pretraining_subjects, evaluation_subjects)

