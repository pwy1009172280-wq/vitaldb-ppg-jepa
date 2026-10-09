"""Stage 4 VitalDB preprocessing: PLETH 500->125 + 16s windows + Solar8000/HR target.

Reads the candidate cohort, parses each case's PLETH + HR tracks, resamples
PLETH to 125 Hz (Kaiser 5.0), windows 16 s, z-scores, computes the HR target per
window, keeps target-eligible windows, hash-orders (vital-window salt), keeps 120
per subject, saves features + targets.
"""

import csv
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, "/gpfs/home2/wpu/projects/vitaldb-ppg-jepa/releases/89915e4fd203111dc5774742893c24333322a652")

from src.experiments.vitaldb_preprocess import compute_hr_target, parse_pleth_csv
from src.preprocessing.transforms import preprocess_run, valid_contiguous_runs

BANK = "/projects/prjs2287/biosignal_bank/datasets/vitaldb"
OUT = "/scratch-shared/wpu/vitaldb_downstream"
WINDOWS_PER_SUBJECT = 120


def sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_hr_csv(path):
    times, values = [], []
    with open(path) as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 2 or p[1].strip() == "":
                continue
            try:
                times.append(float(p[0]))
                values.append(float(p[1]))
            except ValueError:
                continue
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def main():
    cohort = read_csv(os.path.join(BANK, "manifest", "vitaldb_candidate_cohort.csv"))
    os.makedirs(OUT, exist_ok=True)

    # group cases by subject
    subj_cases = {}
    for r in cohort:
        subj_cases.setdefault(r["subjectid"], []).append(r)

    n_ok = 0
    n_short = 0
    for subj, cases in subj_cases.items():
        wins = []  # (hash, ident, window, target)
        for c in cases:
            pleth_path = os.path.join(BANK, "data", "tracks", "pleth", c["snuadc_pleth_tid"] + ".csv")
            hr_path = os.path.join(BANK, "data", "tracks", "hr", c["solar8000_hr_tid"] + ".csv")
            if not (os.path.exists(pleth_path) and os.path.exists(hr_path)):
                continue
            try:
                sig, fs = parse_pleth_csv(pleth_path)
                hr_t, hr_v = parse_hr_csv(hr_path)
            except Exception:
                continue
            for a, b in valid_contiguous_runs(sig, fs):
                run = sig[a:b]
                if run.size < int(round(20 * fs)):
                    continue
                res = preprocess_run(run, fs, native_rate=fs)
                for k, w in enumerate(res.windows):
                    # res.start_samples[k] is in resampled (125 Hz) samples; map to
                    # native case-time: run_start + (trim + k*16s) in seconds.
                    win_start = a / fs + res.start_samples[k] / 125.0
                    t = compute_hr_target(hr_t, hr_v, win_start)
                    if t is None:
                        continue
                    target, coverage, n_upd, span = t
                    ident = f"{subj}|{c['caseid']}|{res.start_samples[k]}"
                    h = sha256(f"first-layerwise-2026-10-09|vital-window|{ident}")
                    wins.append((h, ident, w, target, coverage))
        wins.sort(key=lambda x: (x[0], x[1]))
        chosen = wins[:WINDOWS_PER_SUBJECT]
        if len(chosen) < WINDOWS_PER_SUBJECT:
            n_short += 1
            continue
        feats = np.stack([x[2] for x in chosen]).astype(np.float32)
        targets = np.asarray([x[3] for x in chosen], dtype=np.float32)
        np.save(os.path.join(OUT, f"features_{subj}.npy"), feats)
        np.save(os.path.join(OUT, f"targets_{subj}.npy"), targets)
        n_ok += 1
        if n_ok % 25 == 0:
            print(f"subjects_ok {n_ok} short {n_short}", flush=True)

    print("STAGE4_DONE subjects_ok", n_ok, "short", n_short)


if __name__ == "__main__":
    main()
