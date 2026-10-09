"""Stage 1 MIMIC pretraining preprocessing: decode -> PLETH -> windows -> z-score -> tensor."""

import csv
import atexit
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile

import numpy as np

CODE_ROOT = os.environ.get("PPG_JEPA_CODE_ROOT")
if CODE_ROOT:
    sys.path.insert(0, CODE_ROOT)

from src.preprocessing.transforms import preprocess_run, valid_contiguous_runs

ARCH = "/projects/prjs2287/biosignal_bank/datasets/mimic3wdb-matched/archives/1.0"
SNAP = "/scratch-shared/wpu/mimic_snapshot"
OUT = "/scratch-shared/wpu/mimic_pretrain_windows"
WINDOWS_PER_SUBJECT = 225
_ARCHIVE_CACHE = {}


def sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def decode_segment(tar_path, member):
    import wfdb

    base = member[:-4] if member.endswith(".hea") else member
    tar_path = os.path.abspath(tar_path)
    if tar_path not in _ARCHIVE_CACHE:
        archive = tarfile.open(tar_path, "r:")
        members = {item.name: item for item in archive.getmembers()}
        _ARCHIVE_CACHE[tar_path] = (archive, members)
    archive, members = _ARCHIVE_CACHE[tar_path]
    with tempfile.TemporaryDirectory(prefix="mimic-seg-") as scratch:
        for name in (member, base + ".dat"):
            info = members.get(name)
            if info is None or not info.isfile():
                raise FileNotFoundError(f"missing regular tar member {name!r} in {tar_path}")
            target = os.path.join(scratch, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with archive.extractfile(info) as source, open(target, "wb") as dest:
                shutil.copyfileobj(source, dest)
        record = wfdb.rdrecord(os.path.join(scratch, base), pn_dir=None)
        names = [str(n) for n in record.sig_name]
        if names.count("PLETH") != 1:
            return None, None
        idx = names.index("PLETH")
        return record.p_signal[:, idx].astype(np.float32), float(record.fs)


def close_archive_cache():
    for archive, _ in _ARCHIVE_CACHE.values():
        archive.close()
    _ARCHIVE_CACHE.clear()


atexit.register(close_archive_cache)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_inputs():
    frozen_path = os.path.join(SNAP, "frozen", "mimic_frozen_cohort.csv")
    frozen_meta_path = os.path.join(SNAP, "frozen", "mimic_frozen_cohort.json")
    with open(frozen_meta_path, encoding="utf-8") as f:
        frozen_meta = json.load(f)
    candidate_path = os.path.join(SNAP, "mimic_candidate_cohort.csv")
    segment_path = os.path.join(SNAP, "mimic_segment_manifest.csv")
    archive_snapshot_path = os.path.join(SNAP, "frozen", "mimic_archive_snapshot.csv")
    if sha256_file(candidate_path) != frozen_meta["candidate_sha256"]:
        raise RuntimeError("MIMIC candidate cohort changed after role freeze")
    if sha256_file(segment_path) != frozen_meta["segment_manifest_sha256"]:
        raise RuntimeError("MIMIC segment manifest changed after role freeze")
    if sha256_file(frozen_path) != frozen_meta["cohort_sha256"]:
        raise RuntimeError("MIMIC frozen cohort hash mismatch")
    if sha256_file(archive_snapshot_path) != frozen_meta["archive_snapshot_sha256"]:
        raise RuntimeError("MIMIC archive source snapshot hash mismatch")
    for archive_row in read_csv(archive_snapshot_path):
        archive_path = archive_row["archive_path"]
        sidecar_path = archive_row["sha256_sidecar_path"]
        if not os.path.isfile(archive_path) or os.path.getsize(archive_path) != int(archive_row["archive_bytes"]):
            raise RuntimeError(f"MIMIC source archive missing or size changed: {archive_path}")
        if os.path.exists(archive_path + ".partial") or sha256_file(sidecar_path) != archive_row["sha256_sidecar_sha256"]:
            raise RuntimeError(f"MIMIC source archive has a partial or changed SHA256 sidecar: {archive_path}")
        with open(sidecar_path, encoding="ascii") as f:
            if f.read().split()[0].lower() != archive_row["expected_sha256"]:
                raise RuntimeError(f"MIMIC source archive SHA256 sidecar content changed: {sidecar_path}")
    if len(read_csv(archive_snapshot_path)) != frozen_meta["archive_shards"]:
        raise RuntimeError("MIMIC frozen archive snapshot shard count mismatch")
    cohort = read_csv(frozen_path)
    counts = {role: sum(row["role"] == role for row in cohort) for role in ("train", "diagnostic")}
    if len(cohort) != 1000 or counts != {"train": 900, "diagnostic": 100}:
        raise RuntimeError(f"invalid frozen cohort composition: {counts}")
    return cohort, frozen_meta


def main():
    cohort, frozen_meta = load_frozen_inputs()
    manifest = read_csv(os.path.join(SNAP, "mimic_segment_manifest.csv"))
    os.makedirs(os.path.join(OUT, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "diagnostic"), exist_ok=True)

    by_subject = {}
    for m in manifest:
        by_subject.setdefault(m["subject_id"], []).append((m["shard"], m["member_path"]))

    qc_rows = []
    sample_rows = []
    n_ok = 0
    n_short = 0
    total_windows = 0
    for row in cohort:
        subj = row["subject_id"]
        role = row["role"]
        members = by_subject.get(subj, [])
        windows = []
        for shard, member in members:
            tar_path = os.path.join(ARCH, member.split("/")[0], shard)
            try:
                sig, fs = decode_segment(tar_path, member)
            except Exception:
                continue
            if sig is None:
                continue
            for a, b in valid_contiguous_runs(sig, fs):
                run = sig[a:b]
                if run.size < int(round(20 * fs)):
                    continue
                res = preprocess_run(run, fs, native_rate=fs)
                for k, w in enumerate(res.windows):
                    start = a + res.start_samples[k]
                    ident = f"{subj}|{member}|{start}"
                    h = sha256(f"first-layerwise-2026-10-09|mimic-window|{ident}")
                    windows.append((h, ident, member, int(start), w))
        windows.sort(key=lambda t: (t[0], t[1]))
        chosen = windows[:WINDOWS_PER_SUBJECT]
        if len(chosen) < WINDOWS_PER_SUBJECT:
            n_short += 1
            qc_rows.append({"subject_id": subj, "role": role, "source_segments": len(members), "eligible_windows": len(windows), "selected_windows": len(chosen), "status": "short", "output_sha256": ""})
            continue
        arr = np.stack([w for _, _, _, _, w in chosen]).astype(np.float32)
        out_path = os.path.join(OUT, role, f"{subj}.npy")
        np.save(out_path, arr)
        arr_sha = sha256_file(out_path)
        for index, (window_hash, ident, member, start, _) in enumerate(chosen):
            sample_rows.append({"subject_id": subj, "role": role, "window_index": index, "sample_identity": ident, "sample_sha256": window_hash, "member_path": member, "start_sample": start})
        qc_rows.append({"subject_id": subj, "role": role, "source_segments": len(members), "eligible_windows": len(windows), "selected_windows": len(chosen), "status": "complete", "output_sha256": arr_sha})
        n_ok += 1
        total_windows += arr.shape[0]
        if n_ok % 50 == 0:
            print(f"subjects_ok {n_ok} short {n_short} windows {total_windows}", flush=True)

    def write_csv(path, rows, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    qc_path = os.path.join(OUT, "mimic_stage1_qc.csv")
    samples_path = os.path.join(OUT, "mimic_pretraining_sample_manifest.csv")
    write_csv(qc_path, qc_rows, ["subject_id", "role", "source_segments", "eligible_windows", "selected_windows", "status", "output_sha256"])
    write_csv(samples_path, sample_rows, ["subject_id", "role", "window_index", "sample_identity", "sample_sha256", "member_path", "start_sample"])
    summary = {
        "schema_version": 1,
        "protocol_id": "first-layerwise-2026-10-09",
        "frozen_cohort_sha256": frozen_meta["cohort_sha256"],
        "source_segment_manifest_sha256": frozen_meta["segment_manifest_sha256"],
        "train_subjects": sum(row["role"] == "train" for row in qc_rows if row["status"] == "complete"),
        "diagnostic_subjects": sum(row["role"] == "diagnostic" for row in qc_rows if row["status"] == "complete"),
        "short_subjects": n_short,
        "windows_per_subject": WINDOWS_PER_SUBJECT,
        "training_windows": sum(row["selected_windows"] for row in qc_rows if row["role"] == "train" and row["status"] == "complete"),
        "diagnostic_windows": sum(row["selected_windows"] for row in qc_rows if row["role"] == "diagnostic" and row["status"] == "complete"),
        "qc_sha256": sha256_file(qc_path),
        "sample_manifest_sha256": sha256_file(samples_path),
    }
    with open(os.path.join(OUT, "mimic_stage1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, sort_keys=True, indent=2)
        f.write("\n")
    print("STAGE1_DONE subjects_ok", n_ok, "short", n_short, "windows", total_windows, flush=True)
    if summary["train_subjects"] != 900 or summary["diagnostic_subjects"] != 100 or n_short:
        raise SystemExit("Stage 1 cohort/window gate failed; see mimic_stage1_qc.csv; Stage 2 must not start")


if __name__ == "__main__":
    main()
