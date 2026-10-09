import csv
import hashlib
import subprocess
import sys

import numpy as np

from scripts.run_formal_pretrain import load_role_windows


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_freeze_mimic_cohort_is_deterministic_and_900_100(tmp_path):
    candidates = tmp_path / "candidate.csv"
    segments = tmp_path / "segments.csv"
    out = tmp_path / "frozen"
    archive_root = tmp_path / "archives"
    (archive_root / "p00").mkdir(parents=True)
    archive_path = archive_root / "p00" / "p00_0000.tar"
    archive_path.write_bytes(b"archive-fixture")
    expected = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (archive_root / "p00" / "p00_0000.tar.sha256").write_text(f"{expected}  p00_0000.tar\n", encoding="ascii")
    candidate_rows = [{"subject_id": f"p{i:06d}", "pleth_segments": 1, "rank": i + 1} for i in range(1000)]
    segment_rows = [{"subject_id": row["subject_id"], "shard": "p00_0000.tar", "member_path": f"p00/{row['subject_id']}/record.hea"} for row in candidate_rows]
    _write_csv(candidates, ["subject_id", "pleth_segments", "rank"], candidate_rows)
    _write_csv(segments, ["subject_id", "shard", "member_path"], segment_rows)
    command = [
        sys.executable, "-m", "scripts.freeze_mimic_cohort",
        "--candidate", str(candidates), "--segments", str(segments), "--output-dir", str(out),
        "--archive-root", str(archive_root),
    ]
    first = subprocess.run(command, text=True, capture_output=True, check=True)
    first_bytes = (out / "mimic_frozen_cohort.csv").read_bytes()
    subprocess.run(command, text=True, capture_output=True, check=True)
    assert (out / "mimic_frozen_cohort.csv").read_bytes() == first_bytes
    assert (out / "mimic_archive_snapshot.csv").is_file()
    assert "TRAIN_SUBJECTS 900" in first.stdout
    assert "DIAGNOSTIC_SUBJECTS 100" in first.stdout
    with open(out / "mimic_frozen_cohort.csv", newline="", encoding="utf-8") as stream:
        roles = [row["role"] for row in csv.DictReader(stream)]
    assert roles.count("train") == 900
    assert roles.count("diagnostic") == 100

    with open(candidates, "a", encoding="utf-8") as stream:
        stream.write("\n")
    rejected = subprocess.run(command, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "refusing to replace existing frozen artifact" in rejected.stderr


def test_pretraining_role_loader_reads_only_train_directory(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "diagnostic").mkdir()
    train = np.zeros((225, 2000), dtype=np.float32)
    diagnostic = np.ones((225, 2000), dtype=np.float32)
    np.save(tmp_path / "train" / "p000001.npy", train)
    np.save(tmp_path / "diagnostic" / "p000002.npy", diagnostic)
    windows, files = load_role_windows(str(tmp_path), "train", 1)
    assert windows.shape == (225, 2000)
    assert np.all(windows == 0)
    assert [path.rsplit("/", 1)[-1] for path in files] == ["p000001.npy"]
