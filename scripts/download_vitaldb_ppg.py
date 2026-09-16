#!/usr/bin/env python3
"""Download raw VitalDB SNUADC/PLETH tracks; no signal preprocessing."""
import argparse, csv, io, json, os, tempfile, time, gzip
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = "https://api.vitaldb.net"
TRACK_NAME = "SNUADC/PLETH"

def fetch(url, retries=3, timeout=60):
    for attempt in range(retries):
        try:
            resp=urlopen(Request(url, headers={"User-Agent": "vitaldb-ppg-jepa/1.0", "Accept-Encoding": "gzip"}), timeout=timeout)
            data=resp.read()
            return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        except (HTTPError, URLError, TimeoutError):
            if attempt + 1 == retries: raise
            time.sleep(2 ** attempt)

def parse_csv(data):
    text=data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))

def filter_tracks(rows, case_id=None):
    out=[r for r in rows if r.get("tname")==TRACK_NAME]
    if case_id is not None: out=[r for r in out if str(r.get("caseid"))==str(case_id)]
    return out

def filename(caseid, tid): return f"case_{int(caseid):04d}__{tid}.csv"

def atomic_download(url, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>0: return "skipped"
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".part",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as dst:
            dst.write(fetch(url)); dst.flush(); os.fsync(dst.fileno())
        os.replace(tmp,path); return "downloaded"
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--metadata-only",action="store_true"); ap.add_argument("--limit",type=int); ap.add_argument("--case-id"); ap.add_argument("--overwrite",action="store_true")
    ap.add_argument("--root",type=Path,default=Path(".")); args=ap.parse_args(); root=args.root
    meta=root/"data/metadata"; raw=root/"data/raw/vitaldb_ppg"; meta.mkdir(parents=True,exist_ok=True); raw.mkdir(parents=True,exist_ok=True)
    rows=parse_csv(fetch(API+"/trks")); selected=filter_tracks(rows,args.case_id)
    (meta/"vitaldb_trks.csv").write_text(io.StringIO().getvalue()) if False else None
    with (meta/"vitaldb_ppg_tracks.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["caseid","tname","tid"]); w.writeheader(); w.writerows({k:r.get(k,"") for k in w.fieldnames} for r in selected)
    print(f"total tracks: {len(rows)}"); print(f"SNUADC/PLETH tracks: {len(selected)}"); print(f"unique cases: {len({r.get('caseid') for r in selected})}")
    if args.metadata_only: return
    chosen=selected[:args.limit] if args.limit is not None else selected
    manifest=meta/"vitaldb_ppg_download_manifest.csv"; fields=["caseid","tid","tname","local_path","download_status"]
    with manifest.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in chosen:
            path=raw/filename(r["caseid"],r["tid"])
            status="downloaded"
            if path.exists() and path.stat().st_size>0 and not args.overwrite: status="skipped"
            else: status=atomic_download(f"{API}/{r['tid']}",path)
            w.writerow({"caseid":r["caseid"],"tid":r["tid"],"tname":r["tname"],"local_path":str(path.relative_to(root)),"download_status":status}); print(f"{status}: {path}")

if __name__=="__main__": main()
