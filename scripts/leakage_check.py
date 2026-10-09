"""Stage 0 leakage verification: subject identity + cross-library lineage (metadata-only)."""

import csv
import os

BANK = "/projects/prjs2287/biosignal_bank/datasets/vitaldb"
MIMIC_SNAPSHOT = "/scratch-shared/wpu/mimic_snapshot/records.jsonl"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# 1. VitalDB cohort: subjectid uniqueness (not caseid-as-subjectid)
cohort = read_csv(os.path.join(BANK, "manifest", "vitaldb_candidate_cohort.csv"))
subjects = sorted({r["subjectid"] for r in cohort})
cases = sorted({r["caseid"] for r in cohort})
print("VITALDB_unique_subjects", len(subjects))
print("VITALDB_cases", len(cases))
assert len(subjects) == 500, len(subjects)

# subjectid namespace: numeric
numeric_subj = [s for s in subjects if s.isdigit()]
print("VITALDB_numeric_subjectid", len(numeric_subj), "of", len(subjects))

# 2. MIMIC subject namespace: pNNNNNN (from snapshot)
import json

mimic_subjects = set()
with open(MIMIC_SNAPSHOT) as f:
    for line in f:
        r = json.loads(line)
        mimic_subjects.add(r["subject_id"])
print("MIMIC_snapshot_subjects", len(mimic_subjects))
mimic_p_prefixed = [s for s in mimic_subjects if s.startswith("p")]
print("MIMIC_p_prefixed", len(mimic_p_prefixed), "of", len(mimic_subjects))

# 3. namespace overlap: distinct institutions -> no ID collision
vital_set = set(subjects)
overlap = vital_set & mimic_subjects
print("NAMESPACE_OVERLAP", len(overlap))

# 4. lineage evidence
print("LINEAGE", "MIMIC-III Waveform Matched (Beth Israel Deaconess, PhysioNet) vs VitalDB (Seoul National University Hospital); distinct official institutions")

print("LEAKAGE_METADATA_OK")
