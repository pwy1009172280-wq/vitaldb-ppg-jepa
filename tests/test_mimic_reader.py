"""MIMIC native reader env-blocks (A1): wfdb missing fails closed, not silently."""

import pytest

from src.data.index import RecordIndexRow, write_jsonl
from src.datasets.mimic3 import MimicPPGReader


def _index(tmp_path):
    row = RecordIndexRow(
        dataset="mimic3wdb-matched", dataset_version="1.0", subject_id="p000020",
        record_id="3544749_0008", segment_id=None, session_id=None, modality="PPG",
        channel_name=("PLETH",), sampling_rate_hz=(125.0,), unit=("UNKNOWN",),
        n_samples=None, duration_s=None, source_path="p00/p000020/3544749_0008.hea",
        source_format="WFDB", source_variant="native", continuity="source-native",
        provenance="PhysioNet MIMIC-III matched", role_reference="registry/role_policy.yaml",
        qc_status="PASS", qc_reason=None,
        subject_identity_status="RESOLVED", subject_identity_namespace="mimic3wdb-matched",
        subject_identity_kind="official_subject_id", subject_source_identity="p000020",
    )
    path = tmp_path / "records.jsonl"
    write_jsonl([row], path)
    return path


def test_mimic_reader_requires_wfdb(tmp_path, monkeypatch):
    path = _index(tmp_path)
    reader = MimicPPGReader(tmp_path, path)
    real_import = __import__

    def import_without_wfdb(name, *args, **kwargs):
        if name == "wfdb":
            raise ImportError("simulated missing wfdb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_wfdb)
    with pytest.raises(ImportError, match="wfdb"):
        reader[0]


def test_mimic_reader_subject_ids_from_resolved_rows(tmp_path):
    path = _index(tmp_path)
    reader = MimicPPGReader(tmp_path, path)
    assert reader.subject_ids() == frozenset({"p000020"})
