"""Selective VitalDB track download for the candidate 500-subject cohort.

Downloads ONLY the SNUADC/PLETH (waveform) and Solar8000/HR (numeric) tracks
for the 522 cases in the candidate cohort, into the authoritative Data Bank.
No full-dataset sync, no PLETH_HR substitution, no caseid-as-subjectid.
"""

import csv
import gzip
import hashlib
import os
import tempfile
import time
import urllib.request

API = "https://api.vitaldb.net"
COHORT = "/projects/prjs2287/biosignal_bank/datasets/vitaldb/manifest/vitaldb_candidate_cohort.csv"
RAW = "/projects/prjs2287/biosignal_bank/datasets/vitaldb/data"
MANIFEST = "/projects/prjs2287/biosignal_bank/datasets/vitaldb/manifest/vitaldb_track_manifest.csv"


def fetch(url, retries=4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vitaldb-ppg-jepa/1.0", "Accept-Encoding": "gzip"})
            data = urllib.request.urlopen(req, timeout=180).read()
            return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise last


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(rel, data):
    path = os.path.join(RAW, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skipped"
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return "downloaded"
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


rows = []
with open(COHORT, newline="") as f:
    rows = list(csv.DictReader(f))

manifest = []
failed = 0
for i, r in enumerate(rows):
    for kind, tid in [("hr", r["solar8000_hr_tid"]), ("pleth", r["snuadc_pleth_tid"])]:
        rel = f"tracks/{kind}/{tid}.csv"
        try:
            data = fetch(f"{API}/{tid}")
            status = atomic_write(rel, data)
            manifest.append({
                "subjectid": r["subjectid"], "caseid": r["caseid"], "kind": kind,
                "tid": tid, "path": rel, "status": status,
                "sha256": sha256(data), "bytes": len(data),
            })
        except Exception as e:
            failed += 1
            manifest.append({
                "subjectid": r["subjectid"], "caseid": r["caseid"], "kind": kind,
                "tid": tid, "path": rel, "status": "failed", "sha256": "", "bytes": 0,
                "error": str(e)[:120],
            })
    if (i + 1) % 25 == 0:
        print(f"progress {i + 1}/{len(rows)} failed={failed}", flush=True)

with open(MANIFEST, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subjectid", "caseid", "kind", "tid", "path", "status", "sha256", "bytes"])
    w.writeheader()
    w.writerows({k: m.get(k, "") for k in w.fieldnames} for m in manifest)

print("DOWNLOAD_DONE cases", len(rows), "tracks", len(manifest), "failed", failed)
