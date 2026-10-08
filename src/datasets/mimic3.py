"""Source-native MIMIC-III Waveform Matched PPG reader (official WFDB semantics).

No online fallback, no hand-rolled format-80 decoder, no fake gain/unit. The
official WFDB library decodes signal name/format/gain/units; this reader only
applies the validated logical-path subject and the explicit channel rule.
"""

from __future__ import annotations

from pathlib import Path

from src.data.base import BaseDataset
from src.data.index import RecordIndexRow, read_jsonl
from src.data.mimic_metadata import parse_logical_path, select_channel_indices
from src.data.samples import SUBJECT_IDENTITY_RESOLVED, UnifiedSample


class MimicPPGReader(BaseDataset):
    """Read a bounded run-local MIMIC subset via the official WFDB library.

    Requires the isolated integration environment with ``wfdb`` installed; a
    missing dependency raises ``ImportError`` (ENVIRONMENT blocker), never a
    fallback decoder.
    """

    name = "mimic3wdb-matched"
    version = "1.0"

    def __init__(self, root: str | Path, index_path: str | Path | None = None, *, channel_rule: str = "PLETH"):
        self.root = Path(root)
        self.index_path = Path(index_path or self.root / "metadata" / "records.jsonl")
        self.rows = tuple(read_jsonl(self.index_path))
        self.channel_rule = channel_rule

    def __len__(self) -> int:
        return len(self.rows)

    def subject_ids(self) -> frozenset[str]:
        return frozenset(
            row.subject_id for row in self.rows
            if row.subject_identity_status == SUBJECT_IDENTITY_RESOLVED and row.subject_id
        )

    def __getitem__(self, index: int) -> UnifiedSample:
        if index < 0 or index >= len(self.rows):
            raise IndexError(index)
        return self.read_record(self.rows[index])

    def read_record(self, row: RecordIndexRow) -> UnifiedSample:
        try:
            import wfdb
        except ImportError as exc:
            raise ImportError(
                "MIMIC native reader requires the isolated integration environment with wfdb installed"
            ) from exc

        ref = parse_logical_path(row.source_path)
        if ref.subject != row.subject_id:
            raise ValueError(
                f"subject mismatch: path subject {ref.subject!r} != index subject {row.subject_id!r}"
            )
        header = self.root / row.source_path
        record = wfdb.rdrecord(str(header.with_suffix("")), pn_dir=None)
        sig_names = tuple(str(name) for name in record.sig_name)
        indices = select_channel_indices(sig_names, self.channel_rule)
        if not indices:
            raise ValueError(
                f"no {self.channel_rule!r} channel in {row.source_path}; decoded names {sig_names}"
            )
        channel = indices[0]
        signal = record.p_signal[:, channel][None, :].astype("float32")
        fs = float(record.fs)
        duration = signal.shape[1] / fs
        units = [str(u) for u in (record.units or [])]
        return UnifiedSample(
            signal=signal,
            subject_id=row.subject_id,
            recording_id=row.record_id,
            dataset=self.name,
            modality="PPG",
            sampling_rate_hz=fs,
            start_time_s=0.0,
            end_time_s=duration,
            channel_names=(sig_names[channel],),
            units=(units[channel] if channel < len(units) else "UNKNOWN",),
            provenance={
                "source_path": row.source_path,
                "source_format": row.source_format,
                "dataset_version": row.dataset_version,
                "reader": type(self).__name__,
                "reader_version": "1.0",
            },
            metadata={"continuity": row.continuity, "role_reference": row.role_reference},
            subject_identity_status=SUBJECT_IDENTITY_RESOLVED,
            subject_identity_namespace=row.subject_identity_namespace or self.name,
            subject_identity_kind="official_subject_id",
            subject_source_identity=ref.subject,
        )
