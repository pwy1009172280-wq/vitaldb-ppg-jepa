import csv
import json
from pathlib import Path

from scripts.data.mimic_ppg_header_inventory import inventory_headers, main, parse_header


def test_parse_header_extracts_channels_and_metadata(tmp_path):
    header = tmp_path / "seg_a.hea"
    header.write_text(
        "seg_a 3 125 250\n"
        "seg_a.dat 16 100(0)/mV 0 16 0 0 0 0 PLETH\n"
        "seg_a.dat 16 100(0)/mV 0 16 0 0 0 0 II\n"
        "seg_a.dat 16 100(0)/mV 0 16 0 0 0 0 ABP\n"
        "# source=test\n"
    )
    parsed = parse_header(header)
    assert parsed.record_id == "seg_a"
    assert parsed.signal_names == ("PLETH", "II", "ABP")
    assert parsed.sampling_frequency == 125
    assert parsed.sample_count == 250
    assert parsed.duration_seconds == 2
    assert parsed.comments == ("source=test",)


def test_inventory_is_header_only_and_reports_flags(tmp_path):
    root = tmp_path / "segment_headers" / "p0001"
    root.mkdir(parents=True)
    (root / "seg_a.hea").write_text(
        "seg_a 2 125 125\n"
        "seg_a.dat 16 100(0)/mV 0 16 0 0 0 0 PLETH\n"
        "seg_a.dat 16 100(0)/mV 0 16 0 0 0 0 ECG\n"
    )
    (root / "seg_b.hea").write_text(
        "seg_b 1 250 500\n"
        "seg_b.dat 16 100(0)/mV 0 16 0 0 0 0 ABP\n"
    )
    rows, failures = inventory_headers(root.parent.parent)
    assert not failures
    assert len(rows) == 2
    assert rows[0]["subject_id"] == "p0001"
    assert rows[0]["has_pleth"] == 1
    assert rows[0]["has_ecg"] == 1
    assert rows[0]["has_abp"] == 0
    assert rows[1]["has_pleth"] == 0
    assert rows[1]["has_abp"] == 1
    assert not (root / "seg_a.dat").exists()


def test_inventory_keeps_parse_failures(tmp_path):
    root = tmp_path / "headers"
    root.mkdir()
    (root / "bad.hea").write_text("not a valid header\n")
    rows, failures = inventory_headers(root)
    assert rows == []
    assert len(failures) == 1
    assert failures[0]["relative_path"] == "bad.hea"


def test_cli_writes_inventory_and_failure_csv(tmp_path, monkeypatch):
    root = tmp_path / "headers"
    root.mkdir()
    (root / "seg.hea").write_text(
        "seg 1 100 100\n"
        "seg.dat 16 100(0)/mV 0 16 0 0 0 0 PLETH\n"
    )
    output = tmp_path / "metadata" / "inventory.csv"
    monkeypatch.setattr("sys.argv", ["inventory", "--headers-root", str(root), "--output", str(output)])
    main()
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert json.loads(rows[0]["signal_names"]) == ["PLETH"]
    assert output.with_name("inventory_failures.csv").exists()
