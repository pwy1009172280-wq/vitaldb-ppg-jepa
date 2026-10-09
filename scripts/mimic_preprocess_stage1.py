"""Stage 1 MIMIC pretraining preprocessing: decode -> PLETH -> windows -> z-score -> tensor."""

import csv
import hashlib
import os
import sys
import tarfile
import tempfile

import numpy as np

sys.path.insert(0, "/gpfs/home2/wpu/projects/vitaldb-ppg-jepa/releases/89915e4fd203111dc5774742893c24333322a652")

from src.preprocessing.transforms import preprocess_run, valid_contiguous_runs

ARCH = "/projects/prjs2287/biosignal_bank/datasets/mimic3wdb-matched/archives/1.0"
SNAP = "/scratch-shared/wpu/mimic_snapshot"
OUT = "/scratch-shared/wpu/mimic_pretrain_windows"
WINDOWS_PER_SUBJECT = 225


def sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def decode_segment(tar_path, member):
    import wfdb

    scratch = tempfile.mkdtemp(prefix="mimic-seg-")
    base = member[:-4] if member.endswith(".hea") else member
    with tarfile.open(tar_path, "r:") as t:
        t.extract(member, scratch)
        t.extract(base + ".dat", scratch)
    record = wfdb.rdrecord(os.path.join(scratch, base), pn_dir=None)
    names = [str(n) for n in record.sig_name]
    if "PLETH" not in names:
        return None, None
    idx = names.index("PLETH")
    return record.p_signal[:, idx].astype(np.float32), float(record.fs)


def main():
    cohort = read_csv(os.path.join(SNAP, "mimic_candidate_cohort.csv"))
    manifest = read_csv(os.path.join(SNAP, "mimic_segment_manifest.csv"))
    os.makedirs(OUT, exist_ok=True)

    by_subject = {}
    for m in manifest:
        by_subject.setdefault(m["subject_id"], []).append((m["shard"], m["member_path"]))

    n_ok = 0
    n_short = 0
    total_windows = 0
    for row in cohort:
        subj = row["subject_id"]
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
                    windows.append((h, ident, w))
        windows.sort(key=lambda t: (t[0], t[1]))
        chosen = windows[:WINDOWS_PER_SUBJECT]
        if len(chosen) < WINDOWS_PER_SUBJECT:
            n_short += 1
            continue
        arr = np.stack([w for _, _, w in chosen]).astype(np.float32)
        np.save(os.path.join(OUT, f"{subj}.npy"), arr)
        n_ok += 1
        total_windows += arr.shape[0]
        if n_ok % 50 == 0:
            print(f"subjects_ok {n_ok} short {n_short} windows {total_windows}", flush=True)

    print("STAGE1_DONE subjects_ok", n_ok, "short", n_short, "windows", total_windows)


if __name__ == "__main__":
    main()
