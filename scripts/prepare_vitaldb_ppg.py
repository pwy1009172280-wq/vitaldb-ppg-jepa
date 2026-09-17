#!/usr/bin/env python3
import argparse,csv,os,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from src.data.ppg_preprocessing import preprocess_ppg_record
FIELDS=['caseid','tid','waveform_path','accepted_index_path','source_windows','accepted_windows','rejected_windows','preprocessing_version','status','error']; VERSION='ppg_preprocessing_v0'
def atomic_npy(path,array):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent); os.close(fd)
    try:
        with open(tmp,'wb') as f: np.save(f,array); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise
def prepare(raw_root,processed_root,source_manifest,output_manifest):
    raw_root,processed_root,output_manifest=map(Path,(raw_root,processed_root,output_manifest)); old={}
    if output_manifest.exists():
        with output_manifest.open(newline='') as f: old={(r['caseid'],r['tid']):r for r in csv.DictReader(f)}
    with Path(source_manifest).open(newline='') as f: sources=list(csv.DictReader(f))
    keys=[(str(s['caseid']),str(s['tid'])) for s in sources]
    if len(keys)!=len(set(keys)):
        seen=set()
        duplicate=next(k for k in keys if k in seen or seen.add(k))
        raise ValueError(f'duplicate source key: caseid={duplicate[0]} tid={duplicate[1]}')
    stems={}
    for s in sources:
        local_path=s.get('local_path') or f"case_{int(s['caseid']):04d}__{s['tid']}.csv"
        path=Path(local_path)
        if path.is_absolute() or len(path.parts)!=1 or path.name in ('','.','..') or path.name!=local_path:
            raise ValueError(f'local_path must be a flat filename under raw_root for caseid={s["caseid"]} tid={s["tid"]}: {local_path!r}')
        stem=path.stem
        if stem in stems and stems[stem]!=(str(s['caseid']),str(s['tid'])): raise ValueError(f'processed output stem collision: {stem!r} for {stems[stem]} and {(str(s["caseid"]),str(s["tid"]))}')
        stems[stem]=(str(s['caseid']),str(s['tid']))
    # The current source manifest is authoritative: retain matching prior rows
    # (including unvisited ones), and remove rows for sources no longer listed.
    state={key:old[key] for key in keys if key in old}
    def persist():
        persist_rows=[state[key] for key in keys if key in state]
        output_manifest.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=output_manifest.name+'.',dir=output_manifest.parent); os.close(fd)
        try:
            with open(tmp,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(persist_rows)
            os.replace(tmp,output_manifest)
        except Exception:
            try: os.unlink(tmp)
            except FileNotFoundError: pass
            raise
    for s in sources:
        key=(str(s['caseid']),str(s['tid'])); raw=raw_root/(s.get('local_path') or f"case_{int(s['caseid']):04d}__{s['tid']}.csv"); stem=raw.stem; wp=processed_root/f'{stem}.npy'; ip=processed_root/f'{stem}__accepted_indices.npy'; prior=old.get(key)
        rel_wp=wp.relative_to(processed_root).as_posix(); rel_ip=ip.relative_to(processed_root).as_posix()
        if prior and prior.get('status')=='complete' and prior.get('preprocessing_version')==VERSION and wp.exists() and ip.exists(): state[key]=dict(prior,waveform_path=rel_wp,accepted_index_path=rel_ip); persist(); continue
        row=dict(caseid=s['caseid'],tid=s['tid'],waveform_path=rel_wp,accepted_index_path=rel_ip,source_windows=0,accepted_windows=0,rejected_windows=0,preprocessing_version=VERSION,status='failed',error='')
        try:
            x=pd.read_csv(raw,usecols=['SNUADC/PLETH'])['SNUADC/PLETH'].to_numpy(dtype=float); ws=preprocess_ppg_record(x,caseid=s['caseid'],tid=s['tid']); good=[w for w in ws if w.quality_pass]
            atomic_npy(wp,np.asarray([w.signal for w in good],dtype=np.float32).reshape((-1,5000))); atomic_npy(ip,np.asarray([w.window_index for w in good],dtype=np.int64)); row.update(source_windows=len(ws),accepted_windows=len(good),rejected_windows=len(ws)-len(good),status='complete')
        except Exception as e: row['error']=f'{type(e).__name__}: {e}'
        state[key]=row; persist()
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--raw-root',required=True); p.add_argument('--processed-root',required=True); p.add_argument('--source-manifest',required=True); p.add_argument('--output-manifest',required=True); a=p.parse_args(); prepare(a.raw_root,a.processed_root,a.source_manifest,a.output_manifest)
