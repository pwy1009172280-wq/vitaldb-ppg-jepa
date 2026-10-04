import json

from src.data.index import RecordIndexRow, read_jsonl, write_jsonl


def test_canonical_index_round_trip_preserves_identity_and_unknowns(tmp_path):
    row = RecordIndexRow(
        dataset="fixture",
        dataset_version="1",
        subject_id="fixture:S1",
        record_id="S1:ecg",
        segment_id=None,
        session_id="S1",
        modality="ECG",
        channel_name=("lead_I",),
        sampling_rate_hz=(500.0,),
        unit=("UNKNOWN",),
        n_samples=None,
        duration_s=None,
        source_path="incoming/S1.hea",
        source_format="WFDB",
        source_variant="native",
        continuity="source-native; no gap repair",
        provenance="fixture source",
        role_reference="registry/role_policy.yaml",
        qc_status="UNKNOWN",
        qc_reason="fixture does not include a waveform",
    )
    path = tmp_path / "records.jsonl"
    assert write_jsonl([row], path) == 1
    loaded = list(read_jsonl(path))
    assert loaded == [row]
    payload = json.loads(path.read_text())
    assert payload["subject_id"] == "fixture:S1"
    assert payload["unit"] == ["UNKNOWN"]


def test_index_rows_are_metadata_only(tmp_path):
    row = RecordIndexRow(
        dataset="fixture", dataset_version="1", subject_id="fixture:S1", record_id="S1",
        segment_id=None, session_id=None, modality="PPG", channel_name=("BVP",),
        sampling_rate_hz=(64.0,), unit=("UNKNOWN",), n_samples=None, duration_s=None,
        source_path="data/data.zip!S1.pkl", source_format="pickle-in-zip",
        source_variant="wrist:BVP", continuity="source-native", provenance="fixture",
        role_reference="registry/role_policy.yaml", qc_status="UNKNOWN",
    )
    assert row.source_format == "pickle-in-zip"
    assert row.n_samples is None
    assert row.duration_s is None
