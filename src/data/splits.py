"""Deterministic subject-level split planning."""

from dataclasses import dataclass
import random
from collections.abc import Iterable

from .base import assert_subject_sets_disjoint


@dataclass(frozen=True)
class SubjectSplit:
    """Disjoint subject assignments for train/validation/test."""

    train: frozenset[str]
    validation: frozenset[str]
    test: frozenset[str]

    def __post_init__(self) -> None:
        assert_subject_sets_disjoint(self.train, self.validation, self.test)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train": sorted(self.train),
            "validation": sorted(self.validation),
            "test": sorted(self.test),
        }

    def all_subjects(self) -> frozenset[str]:
        return self.train | self.validation | self.test


def plan_subject_split(
    subject_ids: Iterable[str],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> SubjectSplit:
    """Create a reproducible grouped split without inspecting sample windows.

    Counts use largest-remainder allocation, so every subject is assigned
    exactly once and small synthetic datasets remain deterministic.
    """
    subjects = list(subject_ids)
    if not subjects or any(not subject for subject in subjects):
        raise ValueError("subject_ids must contain at least one non-empty id")
    if len(set(subjects)) != len(subjects):
        raise ValueError("subject_ids must be unique")
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must contain three non-negative values")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must sum to 1")

    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    raw_counts = [len(shuffled) * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    for index in sorted(range(3), key=lambda i: raw_counts[i] - counts[i], reverse=True):
        if sum(counts) >= len(shuffled):
            break
        counts[index] += 1

    train_end = counts[0]
    validation_end = train_end + counts[1]
    return SubjectSplit(
        train=frozenset(shuffled[:train_end]),
        validation=frozenset(shuffled[train_end:validation_end]),
        test=frozenset(shuffled[validation_end:]),
    )

