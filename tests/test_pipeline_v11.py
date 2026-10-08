import numpy as np
import pytest

from src.data import (
    DatasetManifest,
    ExperimentManifest,
    SubjectLeakageError,
    assert_sample_splits_disjoint,
    find_subject_overlap,
    plan_subject_split,
)
from src.data.samples import SUBJECT_IDENTITY_RESOLVED, UnifiedSample
from src.preprocessing import PreprocessingRegistry, TransformNotFoundError


def synthetic_sample(subject: str, value: float = 0.0) -> UnifiedSample:
    return UnifiedSample(
        signal=np.full((1, 4), value, dtype=np.float32),
        subject_id=subject,
        recording_id=f"synthetic-{subject}",
        dataset="synthetic",
        modality="ECG",
        sampling_rate_hz=100.0,
        start_time_s=0.0,
        end_time_s=0.04,
        subject_identity_status=SUBJECT_IDENTITY_RESOLVED,
        subject_identity_namespace="synthetic",
    )


def test_subject_planner_is_reproducible_and_complete():
    subjects = [f"s{i}" for i in range(20)]
    first = plan_subject_split(subjects, seed=11)
    second = plan_subject_split(subjects, seed=11)
    assert first == second
    assert first.all_subjects() == set(subjects)
    assert not (first.train & first.validation)
    assert not (first.train & first.test)
    assert not (first.validation & first.test)


def test_subject_planner_rejects_duplicates_and_bad_ratios():
    with pytest.raises(ValueError, match="unique"):
        plan_subject_split(["s1", "s1"])
    with pytest.raises(ValueError, match="sum"):
        plan_subject_split(["s1"], ratios=(0.5, 0.5, 0.5))


def test_leakage_utility_reports_named_overlap():
    overlap = find_subject_overlap({"train": {"s1", "s2"}, "test": {"s2"}})
    assert overlap == {"s2": ("train", "test")}
    with pytest.raises(SubjectLeakageError, match="s2"):
        assert_sample_splits_disjoint(
            {"train": [synthetic_sample("s2")], "test": [synthetic_sample("s2")]}
        )


def test_manifests_are_metadata_only_and_serializable():
    dataset = DatasetManifest("synthetic", "v1", "ECG", ("s1", "s2"))
    split = plan_subject_split(dataset.subject_ids, ratios=(0.5, 0.0, 0.5), seed=0)
    experiment = ExperimentManifest("smoke", "ssl_pretrain", (dataset.name,), split, seed=0)
    assert dataset.to_dict()["subject_ids"] == ["s1", "s2"]
    assert experiment.to_dict()["split"] == split.as_dict()


def test_preprocessing_registry_composes_registered_synthetic_transform():
    registry = PreprocessingRegistry()
    registry.register("add_one", lambda sample: synthetic_sample(sample.subject_id, sample.signal[0, 0] + 1))
    transformed = registry.compose(["add_one"])(synthetic_sample("s1"))
    assert transformed.signal[0, 0] == pytest.approx(1.0)
    assert registry.names() == ("add_one",)
    with pytest.raises(TransformNotFoundError):
        registry.get("missing")
    with pytest.raises(ValueError, match="already"):
        registry.register("add_one", lambda sample: sample)
