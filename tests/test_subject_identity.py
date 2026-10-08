"""Deterministic test vectors for the subject identity carrier contract."""

import numpy as np
import pytest

from src.data import SubjectIdentityError, UnifiedSample, assert_subject_disjoint
from src.data.index import RecordIndexRow
from src.data.samples import (
    SUBJECT_IDENTITY_RESOLVED,
    SUBJECT_IDENTITY_UNRESOLVED,
)


def sample(subject_id, *, status=SUBJECT_IDENTITY_RESOLVED):
    return UnifiedSample(
        signal=np.ones((1, 4), dtype=np.float32),
        subject_id=subject_id,
        recording_id="r",
        dataset="synthetic",
        modality="PPG",
        sampling_rate_hz=100.0,
        start_time_s=0.0,
        end_time_s=0.04,
        subject_identity_status=status,
        subject_identity_namespace="synthetic" if status == SUBJECT_IDENTITY_RESOLVED else None,
        subject_source_identity="raw-1",
    )


def test_resolved_requires_non_empty_subject_id():
    assert sample("s1").subject_id == "s1"
    with pytest.raises(ValueError, match="subject_id"):
        sample("", status=SUBJECT_IDENTITY_RESOLVED)
    with pytest.raises(ValueError, match="subject_id"):
        sample(None, status=SUBJECT_IDENTITY_RESOLVED)


def test_unresolved_requires_none_subject_id():
    with pytest.raises(ValueError, match="subject_id"):
        UnifiedSample(
            signal=np.ones((1, 4), dtype=np.float32), subject_id="legacy-id",
            recording_id="r", dataset="d", modality="PPG", sampling_rate_hz=100.0,
            start_time_s=0.0, end_time_s=0.04,
        )  # default status UNRESOLVED + non-None subject_id -> rejected


def test_invalid_status_rejected():
    with pytest.raises(ValueError, match="subject_identity_status"):
        UnifiedSample(
            signal=np.ones((1, 4), dtype=np.float32), subject_id=None,
            recording_id="r", dataset="d", modality="PPG", sampling_rate_hz=100.0,
            start_time_s=0.0, end_time_s=0.04, subject_identity_status="BOGUS",
        )


def test_unresolved_sample_can_carry_raw_source_identity():
    s = UnifiedSample(
        signal=np.ones((1, 4), dtype=np.float32), subject_id=None,
        recording_id="r", dataset="d", modality="PPG", sampling_rate_hz=100.0,
        start_time_s=0.0, end_time_s=0.04,
        subject_source_identity="case-123",
    )
    assert s.subject_id is None
    assert s.subject_identity_status == SUBJECT_IDENTITY_UNRESOLVED
    assert s.subject_source_identity == "case-123"


def test_subject_disjoint_rejects_unresolved():
    with pytest.raises(SubjectIdentityError, match="RESOLVED"):
        assert_subject_disjoint([sample("s1"), sample(None, status=SUBJECT_IDENTITY_UNRESOLVED)], [sample("s2")])


def test_index_row_soft_invariant_keeps_legacy_identifier():
    row = RecordIndexRow(
        dataset="d", dataset_version="1", subject_id="legacy:1", record_id="r",
        segment_id=None, session_id=None, modality="ECG", channel_name=("I",),
        sampling_rate_hz=(100.0,), unit=("uV",), n_samples=None, duration_s=None,
        source_path="r.hea", source_format="WFDB", source_variant="native",
        continuity="c", provenance="p", role_reference="role", qc_status="PASS",
    )
    # legacy rows default to UNRESOLVED but keep the legacy subject_id string
    assert row.subject_identity_status == SUBJECT_IDENTITY_UNRESOLVED
    assert row.subject_id == "legacy:1"


def test_index_row_resolved_requires_non_empty():
    with pytest.raises(ValueError, match="subject_id"):
        RecordIndexRow(
            dataset="d", dataset_version="1", subject_id=None, record_id="r",
            segment_id=None, session_id=None, modality="ECG", channel_name=("I",),
            sampling_rate_hz=(100.0,), unit=("uV",), n_samples=None, duration_s=None,
            source_path="r.hea", source_format="WFDB", source_variant="native",
            continuity="c", provenance="p", role_reference="role", qc_status="PASS",
            subject_identity_status=SUBJECT_IDENTITY_RESOLVED,
        )


def test_valid_mask_allows_nonfinite_only_at_invalid_positions():
    sig = np.ones((1, 4), dtype=np.float32)
    sig[0, 0] = np.nan
    # no mask -> any non-finite rejected
    with pytest.raises(ValueError, match="finite"):
        UnifiedSample(
            signal=sig, subject_id=None, recording_id="r", dataset="d", modality="PPG",
            sampling_rate_hz=100.0, start_time_s=0.0, end_time_s=0.04,
        )
    # mask marks position 0 invalid -> allowed
    s = UnifiedSample(
        signal=sig, subject_id=None, recording_id="r", dataset="d", modality="PPG",
        sampling_rate_hz=100.0, start_time_s=0.0, end_time_s=0.04,
        valid_mask=np.array([False, True, True, True]),
    )
    assert not s.valid_mask[0]
    # non-finite at a VALID position is rejected even with a mask
    sig2 = np.ones((1, 4), dtype=np.float32)
    sig2[0, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        UnifiedSample(
            signal=sig2, subject_id=None, recording_id="r", dataset="d", modality="PPG",
            sampling_rate_hz=100.0, start_time_s=0.0, end_time_s=0.04,
            valid_mask=np.array([False, True, True, True]),
        )
