"""Explicit leakage checks for manifests, samples, and planned splits."""

from collections.abc import Iterable, Mapping

from .base import SubjectIdentityError, SubjectLeakageError, assert_subject_sets_disjoint
from .samples import UnifiedSample


def _resolved_only(subject_ids: Iterable[str], what: str) -> list[str]:
    out = []
    for subject in subject_ids:
        if subject is None or not isinstance(subject, str) or not subject:
            raise SubjectIdentityError(f"{what} contains an unresolved/empty subject")
        out.append(subject)
    return out


def find_subject_overlap(splits: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    """Return each subject found in more than one named split."""
    locations: dict[str, list[str]] = {}
    for split_name, subject_ids in splits.items():
        for subject_id in set(_resolved_only(subject_ids, split_name)):
            locations.setdefault(subject_id, []).append(split_name)
    return {
        subject_id: tuple(names)
        for subject_id, names in locations.items()
        if len(names) > 1
    }


def assert_manifest_splits_disjoint(splits: Mapping[str, Iterable[str]]) -> None:
    """Validate named subject sets and report the exact overlapping subjects."""
    overlap = find_subject_overlap(splits)
    if overlap:
        details = ", ".join(f"{subject}: {names}" for subject, names in sorted(overlap.items()))
        raise SubjectLeakageError(f"subject leakage detected ({details})")
    assert_subject_sets_disjoint(*splits.values())


def assert_sample_splits_disjoint(splits: Mapping[str, Iterable[UnifiedSample]]) -> None:
    """Validate sample collections by their subject identity only."""
    assert_manifest_splits_disjoint(
        {name: (sample.subject_id for sample in samples) for name, samples in splits.items()}
    )


def assert_pretraining_downstream_disjoint(
    pretraining_subjects: Iterable[str], downstream_subjects: Iterable[str]
) -> None:
    """Prevent representation pretraining from seeing downstream subjects."""
    pretrain = _resolved_only(pretraining_subjects, "pretraining_subjects")
    downstream = _resolved_only(downstream_subjects, "downstream_subjects")
    overlap = set(pretrain).intersection(downstream)
    if overlap:
        raise SubjectLeakageError(
            f"pretraining/downstream subject overlap detected: {sorted(overlap)!r}"
        )
