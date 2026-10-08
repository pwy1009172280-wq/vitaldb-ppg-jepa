#!/usr/bin/env python3
"""Metadata-first, resumable MIMIC PPG inventory and download pipeline.

Only headers and explicitly selected signal files are handled. Records remain
separate; this tool never creates synthetic continuous sequences.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, tempfile, time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

INV = ["kind","source_relpath","subject_id","record_id","url","local_path","status","size_bytes","sha256","error"]
MAN = ["subject_id","record_id","segment_id","record_relpath","header_path","pleth_presence","sampling_rate","n_samples","duration_seconds","continuity_information","required_remote_files","estimated_bytes","status","error"]

def lines(path): return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
def subject(rel):
    p = rel.strip("/").split("/"); return p[1] if len(p) > 1 else ""
def rid(rel): return rel.strip("/").split("/")[-1]
def local_for(root, kind, url):
    rel = urlparse(url).path.split("/mimic3wdb-matched/1.0/", 1)[-1].lstrip("/")
    return Path(root)/("headers_v2" if kind == "waveform" else "layouts")/rel
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def write_csv(path, fields, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",dir=path.parent); Path(tmp).unlink(missing_ok=True)
    try:
        with open(tmp,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows); f.flush()
        Path(tmp).replace(path)
    finally: Path(tmp).unlink(missing_ok=True)

def inventory(a):
    root=Path(a.root); rec=lines(root/"RECORDS-waveforms"); urls=lines(root/"waveform_header_urls.txt"); layouts=lines(root/"layout_header_urls.txt")
    if len(rec)!=len(urls): raise SystemExit("waveform records and URLs differ")
    rows=[]
    for rel,url in zip(rec,urls):
        p=local_for(root,"waveform",url); ok=p.is_file() and p.stat().st_size>0
        rows.append(dict(kind="waveform",source_relpath=rel,subject_id=subject(rel),record_id=rid(rel),url=url,local_path=str(p.relative_to(root)),status="downloaded" if ok else "missing",size_bytes=p.stat().st_size if ok else "",sha256=sha(p) if ok else "",error=""))
    for url in layouts:
        rel=urlparse(url).path.split("/mimic3wdb-matched/1.0/",1)[-1]; p=local_for(root,"layout",url); ok=p.is_file() and p.stat().st_size>0
        rows.append(dict(kind="layout",source_relpath=rel,subject_id=subject(rel),record_id=Path(rel).stem,url=url,local_path=str(p.relative_to(root)),status="downloaded" if ok else "missing",size_bytes=p.stat().st_size if ok else "",sha256=sha(p) if ok else "",error=""))
    write_csv(root/a.output,INV,rows)
    report=dict(waveform_records=len(rec),waveform_headers=sum(x["kind"]=="waveform" and x["status"]=="downloaded" for x in rows),layout_urls=len(layouts),layout_headers=sum(x["kind"]=="layout" and x["status"]=="downloaded" for x in rows),waveform_data_downloaded=False)
    (root/a.report).write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

def fetch(url,dest):
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_file() and dest.stat().st_size: return "downloaded",dest.stat().st_size,sha(dest)
    part=dest.with_name(dest.name+".part"); offset=part.stat().st_size if part.exists() else 0
    try:
        with urlopen(Request(url,headers={"Range":f"bytes={offset}-"} if offset else {}),timeout=60) as r:
            if offset and r.status != 206: offset=0
            with part.open("ab" if offset else "wb") as f:
                while b:=r.read(1<<20): f.write(b)
        part.replace(dest); return "downloaded",dest.stat().st_size,sha(dest)
    except Exception as e: return "error","",f"{type(e).__name__}: {e}"

def download_headers(a):
    if a.max_files<=0: raise SystemExit("--max-files must be positive")
    root=Path(a.root)
    with (root/a.inventory).open(newline="") as f: rows=list(csv.DictReader(f))
    chosen=[r for r in rows if a.kind=="both" or r["kind"]==a.kind]; chosen=[r for r in chosen if r["status"]!="downloaded"][:a.max_files]
    for i,r in enumerate(chosen,1):
        st,size,detail=fetch(r["url"],root/r["local_path"]); r.update(status=st,size_bytes=size,error="" if st=="downloaded" else detail); r["sha256"]=detail if st=="downloaded" else r.get("sha256",""); print(f"{i}/{len(chosen)} {r['kind']} {r['record_id']} {st}"); time.sleep(a.sleep)
    write_csv(root/a.inventory,INV,rows)

def parse_header(path):
    ls=[x.strip() for x in Path(path).read_text(errors="replace").splitlines() if x.strip()]; first=ls[0].split()
    if len(first)<3: raise ValueError("invalid WFDB header")
    n=int(first[1]); names=[]; files=[]; sizes=[]
    for line in ls[1:1+n]:
        f=line.split();
        if len(f)>=9: files.append(f[0]); names.append(f[8])
        elif f: files.append(f[0]); names.append(f[-1])
        sizes.append(0)
    fs=first[2].split("/")[0]; samples=first[3] if len(first)>3 else ""
    return dict(n=n,fs=fs,samples=samples,names=names,files=files),ls

def manifest(a):
    root=Path(a.root); rx=re.compile(a.pleth_regex,re.I) if a.pleth_regex else None
    with (root/a.inventory).open(newline="") as f: inv=list(csv.DictReader(f))
    rows=[]
    for r in inv:
        if r["kind"]!="waveform": continue
        base=dict(subject_id=r["subject_id"],record_id=r["record_id"],segment_id=r["record_id"],record_relpath=r["source_relpath"],header_path=r["local_path"],pleth_presence="unresolved",sampling_rate="",n_samples="",duration_seconds="",continuity_information="unresolved;record-boundary-preserved",required_remote_files=r["url"],estimated_bytes="",status=r["status"],error=r.get("error",""))
        if r["status"]!="downloaded": rows.append(base); continue
        try:
            h,_=parse_header(root/r["local_path"]); matches=[n for n in h["names"] if rx and rx.search(n)]
            base.update(pleth_presence=("present" if len(matches)==1 else "ambiguous" if matches else "absent") if rx else "unresolved",sampling_rate=h["fs"],n_samples=h["samples"],duration_seconds=(float(h["samples"])/float(h["fs"]) if h["samples"] else ""),required_remote_files=";".join(sorted(set(h["files"]))))
            base["status"]="header_parsed"
        except Exception as e: base.update(status="parse_error",error=f"{type(e).__name__}: {e}")
        rows.append(base)
    write_csv(root/a.output,MAN,rows); print(json.dumps(dict(rows=len(rows),pleth_selection=bool(rx),waveform_data_downloaded=False),indent=2))

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True); common=dict()
    q=s.add_parser("inventory"); q.add_argument("--root",required=True); q.add_argument("--output",default="mimic_inventory.csv"); q.add_argument("--report",default="mimic_inventory.json"); q.set_defaults(fn=inventory)
    q=s.add_parser("download-headers"); q.add_argument("--root",required=True); q.add_argument("--inventory",default="mimic_inventory.csv"); q.add_argument("--kind",choices=["waveform","layout","both"],default="layout"); q.add_argument("--max-files",type=int,required=True); q.add_argument("--sleep",type=float,default=0); q.set_defaults(fn=download_headers)
    q=s.add_parser("manifest"); q.add_argument("--root",required=True); q.add_argument("--inventory",default="mimic_inventory.csv"); q.add_argument("--output",default="mimic_ppg_manifest.csv"); q.add_argument("--pleth-regex",default=None); q.set_defaults(fn=manifest)
    a=p.parse_args(); a.fn(a)
if __name__=="__main__": main()
