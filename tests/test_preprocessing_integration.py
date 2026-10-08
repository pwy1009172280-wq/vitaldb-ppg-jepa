"""Sequential fitted-preprocessing correctness (A4)."""

from dataclasses import replace

import numpy as np

from src.data import SplitContext, UnifiedSample
from src.data.samples import SUBJECT_IDENTITY_RESOLVED
from src.preprocessing import FittedTransform, PreprocessingPipeline


def sample(subject, value):
    return UnifiedSample(
        signal=np.full((1, 4), value, dtype=np.float32),
        subject_id=subject,
        recording_id=f"r-{subject}",
        dataset="synthetic", modality="PPG", sampling_rate_hz=100.0,
        start_time_s=0.0, end_time_s=0.04,
        subject_identity_status=SUBJECT_IDENTITY_RESOLVED,
        subject_identity_namespace="synthetic",
    )


class ScaleBy2(FittedTransform):
    def _fit(self, samples):
        self.fitted_on = [float(s.signal[0, 0]) for s in samples]

    def _apply(self, sample):
        return replace(sample, signal=sample.signal * 2)


class RecordMax(FittedTransform):
    def _fit(self, samples):
        self.max_seen = max(float(s.signal.max()) for s in samples)

    def _apply(self, sample):
        return sample


def test_second_fitted_transform_sees_first_output():
    pipeline = PreprocessingPipeline((ScaleBy2(), RecordMax()))
    ctx = SplitContext("train", frozenset({"s"}))
    pipeline.fit([sample("s", 1.0), sample("s", 3.0)], context=ctx)
    # sequential: RecordMax sees [2, 6] -> 6; the old bug would see [1, 3] -> 3
    assert pipeline.transforms[1].max_seen == 6.0


def test_fit_state_is_exposed_in_metadata():
    pipeline = PreprocessingPipeline((ScaleBy2(), RecordMax()))
    ctx = SplitContext("train", frozenset({"s"}))
    pipeline.fit([sample("s", 1.0)], context=ctx)
    meta = pipeline.metadata()
    assert meta[0]["fitted"] is True
    assert meta[1]["state"]["max_seen"] == 2.0


def test_apply_does_not_refit_or_change_state():
    pipeline = PreprocessingPipeline((ScaleBy2(), RecordMax()))
    ctx = SplitContext("train", frozenset({"s"}))
    pipeline.fit([sample("s", 1.0)], context=ctx)
    before = pipeline.transforms[1].max_seen
    out = pipeline(sample("s", 100.0))
    assert pipeline.transforms[1].max_seen == before
    assert out.signal[0, 0] == 200.0


def test_fit_requires_train_context():
    pipeline = PreprocessingPipeline((ScaleBy2(),))
    ctx = SplitContext("validation", frozenset({"s"}))
    try:
        pipeline.fit([sample("s", 1.0)], context=ctx)
    except ValueError as error:
        assert "train" in str(error)
    else:
        raise AssertionError("fit with validation context should fail")
