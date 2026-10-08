"""Canonical subject/sample identity helpers — single authority.

All subject/sample digests in the pipeline (cache keys, task alignment, result
provenance) delegate here so identity semantics have exactly one implementation.
Hashes are computed with ``src.provenance.content_hash`` (Serialization v1).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.provenance import content_hash

from .samples import SUBJECT_IDENTITY_RESOLVED, UnifiedSample


def canonical_subject(
    subject_id: str | None,
    *,
    namespace: str | None = None,
    kind: str | None = None,
    source_identity: str | None = None,
    mapping_ref: str | None = None,
) -> dict[str, Any]:
    """Return the canonical subject descriptor (status is a separate field)."""
    return {
        "subject_id": subject_id,
        "namespace": namespace,
        "kind": kind,
        "source_identity": source_identity,
        "mapping_ref": mapping_ref,
    }


def subject_identity_hash(
    subject_id: str,
    *,
    namespace: str,
    kind: str | None = None,
    source_identity: str | None = None,
    mapping_ref: str | None = None,
) -> str:
    """Hash of a RESOLVED subject identity (subject_id must be non-empty)."""
    if not isinstance(subject_id, str) or not subject_id:
        raise ValueError("subject_identity_hash requires a non-empty RESOLVED subject_id")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("subject_identity_hash requires a non-empty namespace")
    return content_hash(
        "subject",
        canonical_subject(
            subject_id, namespace=namespace, kind=kind,
            source_identity=source_identity, mapping_ref=mapping_ref,
        ),
    )


def sample_identity_hash(sample: UnifiedSample) -> str:
    """Canonical identity digest of one sample.

    Includes dataset, source/version evidence carried in provenance, the subject
    carrier and resolution status, record identity, ordered channels/units,
    native interval and timebase, continuity, and the native validity mask.
    A source-content or mask change therefore changes the identity; a pure path
    relocation does not.
    """
    mask = np.asarray(sample.valid_mask) if sample.valid_mask is not None else None
    return content_hash(
        "sample",
        {
            "dataset": sample.dataset,
            "source_version": sample.provenance.get("dataset_version"),
            "source_format": sample.provenance.get("source_format"),
            "source_variant": sample.provenance.get("source_variant"),
            "source_identity": sample.provenance.get("source_identity"),
            "subject": canonical_subject(
                sample.subject_id,
                namespace=sample.subject_identity_namespace,
                kind=sample.subject_identity_kind,
                source_identity=sample.subject_source_identity,
                mapping_ref=sample.subject_identity_mapping_ref,
            ),
            "subject_status": sample.subject_identity_status,
            "recording_id": sample.recording_id,
            "channels": list(sample.channel_names),
            "units": list(sample.units),
            "sampling_rate_hz": sample.sampling_rate_hz,
            "native_interval": [sample.window_start_sample, sample.window_end_sample],
            "start_time_s": sample.start_time_s,
            "end_time_s": sample.end_time_s,
            "continuity": sample.metadata.get("continuity"),
            "valid_mask": mask,
        },
    )


def ordered_sample_set_hash(samples: Sequence[UnifiedSample]) -> str:
    """Hash of an ordered, duplicate-free sample identity list.

    Order is semantic. Duplicate sample identities fail closed (never silently
    deduplicated).
    """
    ids: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        identity = sample_identity_hash(sample)
        if identity in seen:
            raise ValueError("duplicate sample identity in ordered sample set")
        seen.add(identity)
        ids.append(identity)
    return content_hash("sample_set", ids)
