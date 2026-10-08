"""Plumbing-only contracts for the generic downstream evaluation path."""

import numpy as np
import pytest

from src.data import UnifiedSample
from src.data.samples import SUBJECT_IDENTITY_RESOLVED
from src.downstream.core import (
    DATA_READY_FOR_EVALUATION,
    READY_FOR_EVALUATION,
    LabelTaskAdapter,
    adapt_targets,
    assess_evaluation_readiness,
)
from src.experiments import EvaluationProtocol


def _sample(label="positive"):
    return UnifiedSample(
        signal=np.ones((1, 8), dtype=np.float32),
        subject_id="subject-1",
        recording_id="recording-1",
        dataset="fixture",
        modality="ppg",
        sampling_rate_hz=100,
        start_time_s=0,
        end_time_s=0.08,
        window_id="window-1",
        labels={"TEST_FIXTURE": label},
        provenance={"source": "TEST_FIXTURE", "annotation_version": "1"},
        subject_identity_status=SUBJECT_IDENTITY_RESOLVED,
        subject_identity_namespace="fixture",
    )


def test_task_adapter_preserves_identity_and_label_provenance():
    adapter = LabelTaskAdapter("TEST_FIXTURE")
    target = adapter.adapt(_sample())
    assert target.value == "positive"
    assert target.sample_id == "fixture/recording-1/window-1"
    assert target.subject_id == "subject-1"
    assert target.provenance["target_name"] == "TEST_FIXTURE"
    assert target.provenance["source_provenance"]["source"] == "TEST_FIXTURE"
    assert adapt_targets((_sample(),), adapter)[0] == target


def test_task_adapter_never_guesses_missing_target():
    with pytest.raises(ValueError, match="missing target label"):
        LabelTaskAdapter("NOT_APPROVED").adapt(_sample())


def test_readiness_separates_data_plumbing_from_protocol():
    data_ready = assess_evaluation_readiness(
        reader_ok=True, labels_ok=True, identities_ok=True, adapter_ok=True,
    )
    assert data_ready.state == DATA_READY_FOR_EVALUATION
    assert data_ready.protocol_blocked
    assert set(data_ready.protocol_blockers) == {"protocol", "target", "split", "metric", "aggregation"}

    protocol = EvaluationProtocol("classification", "linear_probe", ("accuracy",), "sample", True)
    ready = assess_evaluation_readiness(
        reader_ok=True, labels_ok=True, identities_ok=True, adapter_ok=True,
        protocol=protocol, target_ref="TEST_FIXTURE", split_ref="fixture-split",
        metric_config=protocol.metrics, aggregation=protocol.aggregation_level,
    )
    assert ready.state == READY_FOR_EVALUATION
    assert ready.ready
