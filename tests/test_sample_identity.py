"""Deterministic vectors for canonical sample/subject identity."""

from dataclasses import replace

import numpy as np
import pytest

from src.data.identity import (
    ordered_sample_set_hash,
    sample_identity_hash,
    subject_identity_hash,
)
from src.data.samples import SUBJECT_IDENTITY_RESOLVED, SUBJECT_IDENTITY_UNRESOLVED, UnifiedSample


def sample(*, subject_id="s1", mask=None, value=0.0, **kwargs):
    return UnifiedSample(
        signal=np.full((1, 4), value, dtype=np.float32),
        subject_id=subject_id,
        recording_id="r1",
        dataset="synthetic",
        modality="PPG",
        sampling_rate_hz=100.0,
        start_time_s=0.0,
        end_time_s=0.04,
        window_start_sample=0,
        window_end_sample=4,
        valid_mask=mask,
        provenance={"dataset_version": "v1", "source_path": "rel/rec.hea"},
        subject_identity_status=SUBJECT_IDENTITY_RESOLVED if subject_id is not None else SUBJECT_IDENTITY_UNRESOLVED,
        subject_identity_namespace="synthetic" if subject_id is not None else None,
        subject_source_identity="raw-1",
        **kwargs,
    )


def test_subject_identity_hash_is_stable_and_kind_scoped():
    a = subject_identity_hash("ppg-bp:1", namespace="ppg-bp", source_identity="1")
    b = subject_identity_hash("ppg-bp:1", namespace="ppg-bp", source_identity="1")
    assert a == b
    assert len(a) == 64
    assert a != subject_identity_hash("ppg-bp:2", namespace="ppg-bp", source_identity="2")


def test_subject_identity_hash_rejects_unresolved():
    with pytest.raises(ValueError, match="subject_id"):
        subject_identity_hash("", namespace="x")
    with pytest.raises(ValueError, match="namespace"):
        subject_identity_hash("s1", namespace="")


def test_sample_identity_relocation_stable_but_window_sensitive():
    base = sample(value=1.0)
    relocated = replace(base, provenance={**base.provenance, "source_path": "elsewhere/rec.hea"})
    # pure path relocation does NOT change identity (path is a locator, not content)
    assert sample_identity_hash(base) == sample_identity_hash(relocated)
    # signal VALUES are not part of identity (source/member digest covers integrity)
    assert sample_identity_hash(base) == sample_identity_hash(sample(value=2.0))
    # a different native window changes identity
    other_window = replace(base, window_start_sample=4, window_end_sample=8)
    assert sample_identity_hash(base) != sample_identity_hash(other_window)


def test_sample_identity_mask_sensitive():
    a = sample(mask=np.array([True, True, True, True]))
    b = sample(mask=np.array([False, True, True, True]))
    assert sample_identity_hash(a) != sample_identity_hash(b)
    c = sample(mask=None)
    assert sample_identity_hash(a) != sample_identity_hash(c)


def test_sample_identity_subject_status_sensitive():
    resolved = sample(subject_id="s1")
    unresolved = sample(subject_id=None)
    assert sample_identity_hash(resolved) != sample_identity_hash(unresolved)


def test_ordered_sample_set_hash_detects_duplicates_and_order():
    a = sample(subject_id="s1")
    b = sample(subject_id="s2")
    assert ordered_sample_set_hash([a, b]) == ordered_sample_set_hash([a, b])
    assert ordered_sample_set_hash([a, b]) != ordered_sample_set_hash([b, a])
    with pytest.raises(ValueError, match="duplicate"):
        ordered_sample_set_hash([a, a])
