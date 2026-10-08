import csv
import json
from pathlib import Path

import numpy as np
import yaml

from src.data import ProcessedPPGUnifiedAdapter
from src.data.processed_dataset import ProcessedPPGDataset
from src.data.samples import SUBJECT_IDENTITY_UNRESOLVED
from scripts.train_unified import run_training
from scripts.run_downstream import run_downstream


def _fixture(tmp_path: Path):
    root = tmp_path / "processed"
    root.mkdir()
    fields = ["caseid", "tid", "waveform_path", "accepted_index_path", "preprocessing_version", "status", "label_target"]
    rows = []
    for index, label in enumerate((0, 1, 0, 1, 0, 1), 1):
        stem = f"case_{index}"
        np.save(root / f"{stem}.npy", np.full((1, 8), index, dtype=np.float32))
        np.save(root / f"{stem}__accepted_indices.npy", np.asarray([0], dtype=np.int64))
        rows.append(dict(caseid=str(index), tid="tid", waveform_path=f"{stem}.npy",
                         accepted_index_path=f"{stem}__accepted_indices.npy",
                         preprocessing_version="fixture-v1", status="complete", label_target=str(label)))
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return root, manifest


def test_processed_ppg_bridge_preserves_identity_and_provenance(tmp_path):
    root, manifest = _fixture(tmp_path)
    legacy = ProcessedPPGDataset(manifest, root, "fixture-v1", 8)
    sample = ProcessedPPGUnifiedAdapter(legacy)[0]
    # VitalDB caseid is a case identity, not a proven patient -> UNRESOLVED
    assert sample.subject_id is None
    assert sample.subject_identity_status == SUBJECT_IDENTITY_UNRESOLVED
    assert sample.subject_source_identity == "1"
    assert sample.recording_id == "tid"
    assert sample.window_id == "1:tid:0"
    assert sample.window_start_sample == 0 and sample.window_end_sample == 8
    assert sample.signal.shape == (1, 8)
    assert sample.labels == {"target": "0"}
    assert sample.provenance["legacy_caseid"] == "1"


def test_round2_daily_use_training_to_downstream_and_disk_reload(tmp_path):
    root, manifest = _fixture(tmp_path)
    train_config = {
        "method": "jepa",
        "model": {"patch_size": 4, "patch_stride": 4, "embed_dim": 8, "depth": 1,
                  "num_heads": 2, "mlp_ratio": 2.0, "dropout": 0.0, "decoder_dim": 4,
                  "decoder_depth": 1, "decoder_num_heads": 2, "decoder_mlp_ratio": 2.0},
        "data": {"manifest": str(manifest), "processed_root": str(root), "input_length": 8,
                 "expected_preprocessing_version": "fixture-v1"},
        "train": {"batch_size": 2, "epochs": 1, "max_updates": 1, "learning_rate": 1e-3,
                  "weight_decay": 0.0, "seed": 3, "amp": False, "num_workers": 0,
                  "pin_memory": False, "drop_last": True, "grad_clip_norm": None,
                  "run_dir": str(tmp_path / "train"), "device": "cpu"},
        "jepa": {"num_target_blocks": 1, "target_block_length": 1, "predictor_dim": 8,
                 "predictor_depth": 1, "predictor_num_heads": 2, "predictor_mlp_ratio": 2.0,
                 "ema_momentum": 0.9, "loss_beta": 1.0},
    }
    train_path = tmp_path / "train.yaml"
    train_path.write_text(yaml.safe_dump(train_config))
    checkpoint = run_training(train_path)
    assert checkpoint.exists()

    downstream_config = {
        "encoder_config": str(train_path), "checkpoint": str(checkpoint),
        "data": {"manifest": str(manifest), "processed_root": str(root),
                 "expected_preprocessing_version": "fixture-v1", "input_length": 8,
                 "subject_resolver": {"1": "fixture-p1", "2": "fixture-p2", "3": "fixture-p3",
                                      "4": "fixture-p4", "5": "fixture-p5", "6": "fixture-p6"},
                 "subject_namespace": "fixture-vitaldb"},
        "split": {"train": ["fixture-p1", "fixture-p2"], "validation": ["fixture-p3", "fixture-p4"],
                  "test": ["fixture-p5", "fixture-p6"]},
        "downstream": {"experiment_name": "daily-smoke", "results_root": str(tmp_path / "results"),
                       "target": "target", "representations": ["final_layer"], "metrics": ["accuracy"],
                       "pooling": "mean", "epochs": 1, "batch_size": 2},
    }
    downstream_path = tmp_path / "downstream.yaml"
    downstream_path.write_text(yaml.safe_dump(downstream_config))
    result = run_downstream(downstream_path)
    run = tmp_path / "results" / "daily-smoke"
    assert result["rows"]
    assert json.loads((run / "metrics.json").read_text())["schema_version"] == 1
    assert (run / "manifest.json").exists()
    assert (run / "evaluation_protocol.json").exists()
    assert (run / "encoder_checkpoint.json").exists()
    assert (run / "feature_cache_manifest.json").exists()
    assert list((run / "predictions").glob("*.npz"))
