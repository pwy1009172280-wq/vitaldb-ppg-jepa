import csv
import gzip
import io
import json
import urllib.request

API = "https://api.vitaldb.net"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "vitaldb-ppg-jepa/1.0", "Accept-Encoding": "gzip"})
    data = urllib.request.urlopen(req, timeout=180).read()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def parse_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


cases = parse_csv(fetch(API + "/cases"))
print("CASES_fields", list(cases[0].keys()))
print("CASES_count", len(cases))
for r in cases[:2]:
    print("CASE", json.dumps(r))

trks = parse_csv(fetch(API + "/trks"))
print("TRKS_fields", list(trks[0].keys()))
print("TRKS_count", len(trks))
hr = [r for r in trks if r.get("tname") == "Solar8000/HR"]
pleth = [r for r in trks if r.get("tname") == "SNUADC/PLETH"]
print("Solar8000HR_count", len(hr), "SNUADCPLETH_count", len(pleth))
print("HR_sample", json.dumps(hr[0]) if hr else "none")

# save metadata to bank
import os
out = "/projects/prjs2287/biosignal_bank/datasets/vitaldb/metadata"
os.makedirs(out, exist_ok=True)
with open(out + "/vitaldb_cases.csv", "w", newline="") as f:
    if cases:
        w = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        w.writeheader()
        w.writerows(cases)
with open(out + "/vitaldb_hr_tracks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["caseid", "tname", "tid"])
    w.writeheader()
    w.writerows({k: r.get(k, "") for k in ["caseid", "tname", "tid"]} for r in hr)
with open(out + "/vitaldb_ppg_tracks.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["caseid", "tname", "tid"])
    w.writeheader()
    w.writerows({k: r.get(k, "") for k in ["caseid", "tname", "tid"]} for r in pleth)
print("METADATA_SAVED", out)
