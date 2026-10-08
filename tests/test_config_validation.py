"""Negative tests for config nested validation and the PPG pretrain role guard."""

import pytest

from src.config import PipelineConfig, load_config


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_role_guard_rejects_vitaldb_and_ecg_pretrain(tmp_path):
    for name in ("vitaldb", "ptb-xl", "mit-bih", "mimic-iv-ecg", "icentia11k"):
        p = _write(tmp_path, f"dataset:\n  pretrain:\n    name: {name}\n    modality: PPG\n")
        with pytest.raises(ValueError, match="forbidden"):
            load_config(p)


def test_role_guard_rejects_non_ppg_modality(tmp_path):
    p = _write(tmp_path, "dataset:\n  pretrain:\n    name: mimic3wdb-matched\n    modality: ECG\n")
    with pytest.raises(ValueError, match="modality"):
        load_config(p)


def test_role_guard_allows_approved_pretrain(tmp_path):
    p = _write(tmp_path, "dataset:\n  pretrain:\n    name: mimic3wdb-matched\n    modality: PPG\n")
    assert load_config(p).dataset["pretrain"]["name"] == "mimic3wdb-matched"


def test_fixture_pretrain_requires_smoke_only(tmp_path):
    p = _write(tmp_path, "dataset:\n  pretrain:\n    name: test-fixture-ppg\n    modality: PPG\n")
    with pytest.raises(ValueError, match="smoke_only"):
        load_config(p)
    p2 = _write(
        tmp_path,
        "dataset:\n  pretrain:\n    name: test-fixture-ppg\n    modality: PPG\n"
        "experiment:\n  smoke_only: true\n",
    )
    assert load_config(p2).dataset["pretrain"]["name"] == "test-fixture-ppg"


def test_unknown_nested_key_rejected(tmp_path):
    p = _write(tmp_path, "dataset:\n  pretrain:\n    name: mimic3wdb-matched\n  bogus: 1\n")
    with pytest.raises(ValueError, match="unknown nested"):
        load_config(p)


def test_pi_placeholder_rejected(tmp_path):
    p = _write(tmp_path, "experiment:\n  split: PI_DECISION_REQUIRED\n")
    with pytest.raises(ValueError, match="PI_DECISION_REQUIRED"):
        load_config(p)


def test_missing_pretrain_name_rejected(tmp_path):
    p = _write(tmp_path, "dataset:\n  pretrain:\n    modality: PPG\n")
    with pytest.raises(ValueError, match="name"):
        load_config(p)
