"""MIMIC candidate cohort: scan archives for PLETH subjects, hash-order, select 1000 (metadata-only)."""

import csv
import glob
import hashlib
import os
import tarfile

ARCH = "/projects/prjs2287/biosignal_bank/datasets/mimic3wdb-matched/archives/1.0"
OUT = "/scratch-shared/wpu/mimic_snapshot"


def scan_tar(path, subjects):
    shard = os.path.basename(path)
    tar = tarfile.open(path, "r:")
    try:
        for m in tar.getmembers():
            if not (m.isfile() and m.name.endswith(".hea") and "_layout" not in m.name and m.name.count("/") >= 2):
                continue
            parts = m.name.split("/")
            subj = parts[1]
            f = tar.extractfile(m)
            if f is None:
                continue
            content = f.read(8192).decode("utf-8", "replace")
            if "PLETH" in content:
                subjects.setdefault(subj, []).append((shard, m.name))
    finally:
        tar.close()


subjects = {}
shards = sorted(glob.glob(f"{ARCH}/p*/p*.tar"))
print("SHARDS", len(shards), flush=True)
for i, shard in enumerate(shards):
    scan_tar(shard, subjects)
    if (i + 1) % 10 == 0:
        print(f"scanned {i + 1}/{len(shards)} subjects={len(subjects)}", flush=True)


def subject_sort_key(subj):
    h = hashlib.sha256(f"first-layerwise-2026-10-09|mimic-subject|{subj}".encode()).hexdigest()
    return (h, subj)


ordered = sorted(subjects.keys(), key=subject_sort_key)
cohort = ordered[:1000]

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "mimic_candidate_cohort.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subject_id", "pleth_segments", "rank"])
    for i, subj in enumerate(cohort):
        w.writerow([subj, len(subjects[subj]), i + 1])

# segment manifest for the 1000 cohort subjects (subject -> shard + member path)
with open(os.path.join(OUT, "mimic_segment_manifest.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subject_id", "shard", "member_path"])
    for subj in cohort:
        for shard, member in sorted(subjects[subj]):
            w.writerow([subj, shard, member])

print("MIMIC_SUBJECTS_WITH_PLETH", len(subjects))
print("MIMIC_CANDIDATE_COHORT", len(cohort))
print("MIMIC_MANIFEST_ROWS", sum(len(subjects[s]) for s in cohort))
print("MIMIC_COHORT_SAVED", os.path.join(OUT, "mimic_candidate_cohort.csv"))
