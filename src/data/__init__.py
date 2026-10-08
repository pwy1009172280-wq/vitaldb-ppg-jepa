"""Dataset contracts and adapters for the biosignal research pipeline."""

from .base import BaseDataset, SubjectIdentityError, SubjectLeakageError, assert_subject_disjoint, assert_subject_sets_disjoint
from .access import SplitAwareDataset, SplitContext
from .leakage import (
    assert_manifest_splits_disjoint,
    assert_pretraining_downstream_disjoint,
    assert_sample_splits_disjoint,
    find_subject_overlap,
)
from .manifests import DatasetManifest, ExperimentManifest
from .samples import UnifiedSample
from .splits import SubjectSplit, plan_subject_split

__all__ = [
    "BaseDataset",
    "SplitAwareDataset",
    "SplitContext",
    "SubjectLeakageError",
    "SubjectIdentityError",
    "UnifiedSample",
    "assert_subject_disjoint",
    "assert_subject_sets_disjoint",
    "assert_manifest_splits_disjoint",
    "assert_pretraining_downstream_disjoint",
    "assert_sample_splits_disjoint",
    "find_subject_overlap",
    "DatasetManifest",
    "ExperimentManifest",
    "SubjectSplit",
    "plan_subject_split",
    "ProcessedPPGUnifiedAdapter",
]


def __getattr__(name):
    if name == "ProcessedPPGUnifiedAdapter":
        from .ppg_adapter import ProcessedPPGUnifiedAdapter
        globals()[name] = ProcessedPPGUnifiedAdapter
        return ProcessedPPGUnifiedAdapter
    raise AttributeError(name)
