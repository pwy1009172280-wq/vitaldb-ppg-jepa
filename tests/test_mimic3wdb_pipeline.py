import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data.mimic3wdb_pipeline import build_manifest, parse_header, safe_delete_raw, validate_processed_output


def test_inventory_selects_pleth_and_preserves_segments(tmp_path):
    header = tmp_path / "layout.hea"
    header.write_text("layout 2 125 250\nlayout_000 16 100(0)/mV 0 16 0 0 0 0 PLETH\nlayout_001 16 100(0)/mV 0 16 0 0 0 0 ECG\nseg_a 125\nseg_b 250\n")
    parsed = parse_header(header)
    assert parsed.sampling_rate == 125
    rows, required, failures = build_manifest(tmp_path)
    assert not failures
    assert len(rows) == 2
    assert {row["segment_record_id"] for row in rows} == {"seg_a", "seg_b"}
    assert all(row["continuity_status"] == "segment_boundary_preserved" for row in rows)
    assert len(required) == 4


def test_safe_delete_requires_valid_provenance(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = raw_root / "segment.dat"
    raw.write_bytes(b"raw")
    output = tmp_path / "processed.npz"
    metadata = {"source_data_path": str(raw.resolve()), "source_sha256": __import__("hashlib").sha256(b"raw").hexdigest()}
    np.savez_compressed(output, signal=np.ones(4, dtype=np.float32), metadata_json=json.dumps(metadata))
    assert validate_processed_output(output)["source_data_path"] == str(raw.resolve())
    safe_delete_raw(raw, output, raw_root)
    assert not raw.exists()


def test_safe_delete_refuses_outside_raw_root(tmp_path):
    raw = tmp_path / "raw.dat"
    raw.write_bytes(b"raw")
    with pytest.raises(ValueError, match="outside raw root"):
        safe_delete_raw(raw, tmp_path / "missing.npz", tmp_path / "raw")
