"""Thin adapter from the frozen ProcessedPPGDataset to the public data contract."""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .base import BaseDataset
from .processed_dataset import ProcessedPPGDataset
from .samples import (
    SUBJECT_IDENTITY_RESOLVED,
    SUBJECT_IDENTITY_UNRESOLVED,
    UnifiedSample,
)


class ProcessedPPGUnifiedAdapter(BaseDataset):
    """Expose processed VitalDB PPG windows without changing preprocessing.

    The legacy dataset remains the file-format reader.  This class only maps
    its row identity and waveform into ``UnifiedSample``.  Sampling rate and
    channel metadata are explicit adapter inputs because the legacy manifest
    does not store them.

    ``caseid`` is a *case* identity, not a proven patient identity.  Unless an
    explicit ``subject_resolver`` (caseid -> canonical patient ID) is supplied,
    samples are emitted with ``subject_id=None`` and
    ``SUBJECT_IDENTITY_UNRESOLVED``; the raw caseid is preserved in
    ``subject_source_identity`` and provenance.
    """

    def __init__(self, dataset: ProcessedPPGDataset, *, sampling_rate_hz: float = 500.0,
                 dataset_name: str = "vitaldb", modality: str = "PPG",
                 channel_names: tuple[str, ...] = ("SNUADC/PLETH",),
                 units: tuple[str, ...] = (), source_provenance: Mapping[str, Any] | None = None,
                 subject_resolver: Callable[[str], str] | None = None,
                 subject_namespace: str | None = None):
        if not isinstance(dataset, ProcessedPPGDataset):
            raise TypeError("dataset must be ProcessedPPGDataset")
        self.dataset = dataset
        self._name = dataset_name
        self.modality = modality
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.channel_names = tuple(channel_names)
        self.units = tuple(units)
        self.source_provenance = dict(source_provenance or {})
        self.subject_resolver = subject_resolver
        self.subject_namespace = subject_namespace
        if self.channel_names and len(self.channel_names) != 1:
            raise ValueError("ProcessedPPGUnifiedAdapter currently exposes one channel")

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> UnifiedSample:
        item = self.dataset[index]
        waveform = item["waveform"].detach().cpu().numpy().astype(np.float32, copy=False)
        caseid, tid, window_index = str(item["caseid"]), str(item["tid"]), int(item["window_index"])
        row_index = int(np.searchsorted(self.dataset.prefix, index, side="right") - 1)
        row = self.dataset.rows[row_index]
        start_sample = window_index * self.dataset.expected_length
        end_sample = start_sample + self.dataset.expected_length
        provenance = {
            "adapter": "ProcessedPPGUnifiedAdapter",
            "manifest": str(self.dataset.manifest),
            "preprocessing_version": self.dataset.expected_preprocessing_version,
            "legacy_caseid": caseid,
            "legacy_tid": tid,
            "legacy_window_index": window_index,
            **self.source_provenance,
        }
        labels = {key.removeprefix("label_"): value for key, value in row.items() if key.startswith("label_") and value != ""}
        if self.subject_resolver is not None:
            subject_id = self.subject_resolver(caseid)
            if not isinstance(subject_id, str) or not subject_id:
                raise ValueError(f"subject_resolver returned invalid subject for caseid={caseid!r}")
            status = SUBJECT_IDENTITY_RESOLVED
            namespace = self.subject_namespace
        else:
            subject_id = None
            status = SUBJECT_IDENTITY_UNRESOLVED
            namespace = None
        return UnifiedSample(
            signal=waveform,
            subject_id=subject_id,
            recording_id=tid,
            dataset=self.name,
            modality=self.modality,
            sampling_rate_hz=self.sampling_rate_hz,
            start_time_s=start_sample / self.sampling_rate_hz,
            end_time_s=end_sample / self.sampling_rate_hz,
            channel_names=self.channel_names,
            units=self.units,
            provenance=provenance,
            labels=labels,
            window_id=f"{caseid}:{tid}:{window_index}",
            window_start_sample=start_sample,
            window_end_sample=end_sample,
            subject_identity_status=status,
            subject_identity_namespace=namespace,
            subject_identity_kind="case_id" if status == SUBJECT_IDENTITY_RESOLVED else "case_id_unresolved",
            subject_source_identity=caseid,
        )
