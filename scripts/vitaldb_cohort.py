import csv
import hashlib
import json
import os

META = "/projects/prjs2287/biosignal_bank/datasets/vitaldb/metadata"
OUT = "/projects/prjs2287/biosignal_bank/datasets/vitaldb/manifest"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


cases = read_csv(os.path.join(META, "vitaldb_cases.csv"))
hr = read_csv(os.path.join(META, "vitaldb_hr_tracks.csv"))
pleth = read_csv(os.path.join(META, "vitaldb_ppg_tracks.csv"))

hr_by_case = {r["caseid"]: r["tid"] for r in hr}
pleth_by_case = {r["caseid"]: r["tid"] for r in pleth}

# subjectid -> list of caseids (resolved, non-empty)
subj_cases = {}
for c in cases:
    sid = (c.get("subjectid") or "").strip()
    if not sid:
        continue
    subj_cases.setdefault(sid, []).append(c["caseid"])

eligible = []
for sid, cids in subj_cases.items():
    good = [(c, hr_by_case[c], pleth_by_case[c]) for c in cids if c in hr_by_case and c in pleth_by_case]
    if good:
        eligible.append((sid, good))


def subject_sort_key(item):
    sid = item[0]
    h = hashlib.sha256(f"first-layerwise-2026-10-09|vital-subject|{sid}".encode()).hexdigest()
    return (h, sid)  # collision -> full id ascending


eligible.sort(key=subject_sort_key)
cohort = eligible[:500]

os.makedirs(OUT, exist_ok=True)
rows = []
for sid, good in cohort:
    for caseid, hrtid, ptid in good:
        rows.append({
            "subjectid": sid, "caseid": caseid,
            "solar8000_hr_tid": hrtid, "snuadc_pleth_tid": ptid,
        })
with open(os.path.join(OUT, "vitaldb_candidate_cohort.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subjectid", "caseid", "solar8000_hr_tid", "snuadc_pleth_tid"])
    w.writeheader()
    w.writerows(rows)

print("eligible_subjects", len(eligible))
print("candidate_cohort_subjects", len(cohort))
print("candidate_cohort_cases", len(rows))
print("cohort_head", json.dumps(rows[:3]))
print("COHORT_SAVED", os.path.join(OUT, "vitaldb_candidate_cohort.csv"))
