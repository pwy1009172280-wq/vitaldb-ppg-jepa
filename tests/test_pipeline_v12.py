from dataclasses import replace

import numpy as np
import pytest

from src.data import (
    DatasetManifest,
    SplitAwareDataset,
    SplitContext,
    SubjectLeakageError,
    UnifiedSample,
    assert_pretraining_downstream_disjoint,
    plan_subject_split,
)
from src.data.base import BaseDataset
from src.data.samples import SUBJECT_IDENTITY_RESOLVED
from src.experiments import EvaluationProtocol, ExperimentRunManifest
from src.preprocessing import FittedTransform, PreprocessingRegistry, StatelessTransform


def synthetic_sample(subject: str, value: float = 0.0) -> UnifiedSample:
    return UnifiedSample(
        signal=np.full((1, 4), value, dtype=np.float32),
        subject_id=subject,
        recording_id=f"recording-{subject}",
        dataset="synthetic",
        modality="ECG",
        sampling_rate_hz=100.0,
        start_time_s=0.0,
        end_time_s=0.04,
        channel_names=("lead_I",),
        units=("mV",),
        valid_mask=np.ones((4,), dtype=bool),
        provenance={"source": "synthetic", "window_index": 0},
        window_id=f"window-{subject}",
        window_start_sample=0,
        window_end_sample=4,
        subject_identity_status=SUBJECT_IDENTITY_RESOLVED,
        subject_identity_namespace="synthetic",
    )


class SyntheticDataset(BaseDataset):
    def __init__(self, samples):
        self.samples = list(samples)

    @property
    def name(self):
        return "synthetic"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class MeanCenter(FittedTransform):
    version = "synthetic-1"

    def __init__(self):
        super().__init__()
        self.fit_subjects = ()

    def _fit(self, samples):
        self.fit_subjects = tuple(sample.subject_id for sample in samples)

    def _apply(self, sample):
        return replace(sample, signal=sample.signal - sample.signal.mean())


class Identity(StatelessTransform):
    def apply(self, sample):
        return sample


def test_unified_sample_has_signal_and_window_provenance():
    sample = synthetic_sample("s1")
    assert sample.channel_names == ("lead_I",)
    assert sample.units == ("mV",)
    assert sample.valid_mask.shape == (4,)
    assert sample.provenance["source"] == "synthetic"
    assert sample.window_end_sample == 4


def test_split_aware_access_requires_context_and_rejects_wrong_subject():
    dataset = SyntheticDataset([synthetic_sample("train"), synthetic_sample("test")])
    with pytest.raises(TypeError):
        SplitAwareDataset(dataset, None)
    view = SplitAwareDataset(dataset, SplitContext("train", frozenset({"train"})))
    assert view[0].subject_id == "train"
    with pytest.raises(SubjectLeakageError):
        view[1]


def test_fitted_preprocessing_can_only_fit_training_subjects():
    transform = MeanCenter()
    validation = SplitContext("validation", frozenset({"validation"}))
    with pytest.raises(ValueError, match="train"):
        transform.fit([synthetic_sample("train")], context=validation)
    with pytest.raises(ValueError, match="outside"):
        transform.fit(
            [synthetic_sample("train"), synthetic_sample("validation")],
            context=SplitContext("train", frozenset({"train"})),
        )
    transform.fit([synthetic_sample("train")], context=SplitContext("train", frozenset({"train"})))
    assert transform.fit_subjects == ("train",)
    assert transform.metadata()["version"] == "synthetic-1"


def test_registry_exposes_explicit_transform_contracts():
    registry = PreprocessingRegistry()
    registry.register("identity", Identity())
    registry.register("center", MeanCenter())
    pipeline = registry.compose(["identity", "center"])
    with pytest.raises(RuntimeError, match="fit"):
        pipeline(synthetic_sample("train"))
    pipeline.fit([synthetic_sample("train")], context=SplitContext("train", frozenset({"train"})))
    assert pipeline.metadata()[1]["fitted"] is True


def test_pretraining_subjects_must_be_disjoint_from_downstream():
    assert_pretraining_downstream_disjoint({"p1"}, {"d1"})
    with pytest.raises(SubjectLeakageError):
        assert_pretraining_downstream_disjoint({"shared"}, {"shared"})


def test_manifest_and_protocol_serialization():
    split = plan_subject_split(["s1", "s2", "s3"], seed=2)
    dataset = DatasetManifest(
        "synthetic", "v2", "ECG", ("s1", "s2", "s3"),
        preprocessing_version="prep-1",
        environment={"python": "test"},
        git_commit="abc123",
        git_state="clean",
    )
    protocol = EvaluationProtocol("heart-rate", "linear_probe", ("mae", "r2"), "subject", True)
    manifest = ExperimentRunManifest(
        "synthetic_eval", {"dataset": dataset.to_dict()}, "dataset://synthetic/v2", "split://2", 2,
        preprocessing_version="prep-1", environment={"python": "test"},
        git_commit="abc123", git_state="clean", evaluation_protocol=protocol,
        pretraining_subject_ids=("pretrain-1",),
    )
    assert dataset.to_dict()["git_state"] == "clean"
    assert manifest.to_dict()["evaluation_protocol"]["protocol_type"] == "linear_probe"
    assert manifest.to_dict()["pretraining_subject_ids"] == ["pretrain-1"]
