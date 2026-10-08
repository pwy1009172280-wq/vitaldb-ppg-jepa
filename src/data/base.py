"""Interfaces and split-safety checks for dataset adapters."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from .samples import UnifiedSample


class SubjectLeakageError(ValueError):
    """Raised when a subject appears in more than one evaluation split."""


class SubjectIdentityError(ValueError):
    """Raised when a subject-aware operation encounters an unresolved subject."""


class BaseDataset(ABC, Sequence[UnifiedSample]):
    """Minimal interface implemented by MIMIC, VitalDB, ECG, and IMU adapters.

    Adapters own file formats and preprocessing orchestration. Consumers only
    depend on this interface and never inspect dataset-specific records.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable dataset identifier used in manifests and experiment logs."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of materialized samples."""

    @abstractmethod
    def __getitem__(self, index: int) -> UnifiedSample:
        """Return one canonical sample."""

    def subject_ids(self) -> frozenset[str]:
        """Return all resolved subjects represented by this dataset instance.

        Fails closed if any sample carries an unresolved (``None``) subject.
        """
        subjects = frozenset(
            sample.subject_id for sample in (self[index] for index in range(len(self)))
        )
        if None in subjects:
            raise SubjectIdentityError(
                f"dataset {self.name!r} contains unresolved subjects; "
                "cannot form a subject set"
            )
        return subjects


def assert_subject_disjoint(*splits: Iterable[UnifiedSample]) -> None:
    """Fail fast if any subject is present in multiple supplied splits."""
    seen: dict[str, int] = {}
    for split_index, samples in enumerate(splits):
        for sample in samples:
            if sample.subject_id is None:
                raise SubjectIdentityError(
                    "subject-aware splits require RESOLVED subjects"
                )
            previous = seen.get(sample.subject_id)
            if previous is not None and previous != split_index:
                raise SubjectLeakageError(
                    f"subject {sample.subject_id!r} occurs in splits "
                    f"{previous} and {split_index}"
                )
            seen[sample.subject_id] = split_index


def assert_subject_sets_disjoint(*subject_sets: Iterable[str]) -> None:
    """Variant for manifests/split planners that have no samples yet."""
    seen: set[str] = set()
    for split_index, subject_set in enumerate(subject_sets):
        current = set(subject_set)
        if None in current or any(not isinstance(subject, str) for subject in current):
            raise SubjectIdentityError(
                "subject sets must contain only non-empty RESOLVED subject strings"
            )
        overlap = seen.intersection(current)
        if overlap:
            raise SubjectLeakageError(
                f"subjects {sorted(overlap)!r} occur in multiple splits; "
                f"latest split index is {split_index}"
            )
        seen.update(current)
