import numpy as np
import pytest

from src.config import PipelineConfig, load_config
from src.data import (
    BaseDataset,
    SubjectLeakageError,
    UnifiedSample,
    assert_subject_disjoint,
    assert_subject_sets_disjoint,
)
from src.data.samples import SUBJECT_IDENTITY_RESOLVED


def sample(subject: str) -> UnifiedSample:
    return UnifiedSample(
        signal=np.zeros((1, 8), dtype=np.float32),
        subject_id=subject,
        recording_id=f"recording-{subject}",
        dataset="synthetic",
        modality="ECG",
        sampling_rate_hz=100.0,
        start_time_s=0.0,
        end_time_s=0.08,
        channel_names=("lead_I",),
        subject_identity_status=SUBJECT_IDENTITY_RESOLVED,
        subject_identity_namespace="synthetic",
    )


class InMemoryDataset(BaseDataset):
    def __init__(self, samples):
        self.samples = list(samples)

    @property
    def name(self):
        return "memory"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_unified_sample_contract_and_dataset_interface():
    dataset = InMemoryDataset([sample("s1")])
    assert dataset.subject_ids() == frozenset({"s1"})
    assert dataset[0].signal.shape == (1, 8)


def test_unified_sample_rejects_bad_shape_and_missing_subject():
    with pytest.raises(ValueError, match="shape"):
        UnifiedSample(np.zeros(8), "s1", "r1", "d", "ECG", 100, 0, 1)
    with pytest.raises(ValueError, match="subject_id"):
        UnifiedSample(np.zeros((1, 8)), "", "r1", "d", "ECG", 100, 0, 1)


def test_subject_disjointness_is_enforced():
    assert_subject_disjoint([sample("train")], [sample("test")])
    with pytest.raises(SubjectLeakageError, match="s1"):
        assert_subject_disjoint([sample("s1")], [sample("s1")])
    with pytest.raises(SubjectLeakageError):
        assert_subject_sets_disjoint({"s1"}, {"s1", "s2"})


def test_config_loader_reads_sections_and_rejects_unknown(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset:\n  pretrain:\n    name: mimic3wdb-matched\n    modality: PPG\n"
        "experiment:\n  seed: 7\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert isinstance(config, PipelineConfig)
    assert config.dataset["pretrain"]["name"] == "mimic3wdb-matched"
    assert config.experiment["seed"] == 7

    path.write_text("training: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)
