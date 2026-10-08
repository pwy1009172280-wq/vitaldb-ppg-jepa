#!/usr/bin/env python3
"""Local-only throughput benchmark. `--prepare` freezes first 200 case IDs."""
from __future__ import annotations
import argparse,csv,json,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.ppg_preprocessing import preprocess_ppg_record
from src.models.ssl_benchmarks import build_method

ROOT=Path(__file__).resolve().parents[1]; CACHE=ROOT/'data/processed/benchmark_200'; RAW=ROOT/'data/raw/vitaldb_ppg'
class Windows(Dataset):
 def __init__(self): self.files=sorted(CACHE.glob('case_*.npy')); self.parts=[]; self.n=0
 def __len__(self): return self.n
 def __getitem__(self,i):
  for start,end,f in self.parts:
   if i<end:return torch.from_numpy(np.load(f,mmap_mode='r')[i-start]).unsqueeze(0)
  raise IndexError(i)
 def load(self):
  for f in self.files:
   a=np.load(f,mmap_mode='r'); self.parts.append((self.n,self.n+len(a),f));self.n+=len(a)
  return self
def prepare(start=0, count=200):
 CACHE.mkdir(parents=True,exist_ok=True); rows=[]; start=time.perf_counter()
 for f in sorted(RAW.glob('case_*.csv'))[:200][args.start:args.start+args.count]:
  caseid=int(f.name.split('_')[1]); tid=f.stem.split('__',1)[1]
  # VitalDB CSV uses one waveform column; blank records become NaN and are screened by v0.
  wave=pd.read_csv(f,usecols=['SNUADC/PLETH'])['SNUADC/PLETH'].to_numpy(dtype=float)
  ws=preprocess_ppg_record(wave,caseid=caseid,tid=tid); good=np.asarray([w.signal for w in ws if w.quality_pass],dtype=np.float32)
  np.save(CACHE/(f.stem+'.npy'),good); rows.append(dict(caseid=caseid,tid=tid,source_filename=f.name,complete_10s_windows=len(ws),rejected_non_finite_windows=sum(w.rejection_reason=='non_finite' for w in ws),rejected_degenerate_windows=sum(w.rejection_reason=='degenerate' for w in ws),accepted_windows=len(good)))
 manifest=CACHE/'manifest.csv'; old=[]
 if manifest.exists(): old=list(csv.DictReader(manifest.open()))
 merged={int(r['caseid']):r for r in old}; merged.update({r['caseid']:r for r in rows}); rows=[merged[k] for k in sorted(merged)]
 if rows:
  with manifest.open('w',newline='') as h:
   w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'cases_processed_this_run':len(rows),'accepted':sum(int(r['accepted_windows']) for r in rows),'rejected':sum(int(r['complete_10s_windows'])-int(r['accepted_windows']) for r in rows),'cache_bytes':sum(f.stat().st_size for f in CACHE.glob('*.npy')),'seconds':time.perf_counter()-start},indent=2))
def train(args):
 if not torch.cuda.is_available(): raise SystemExit('CUDA is unavailable; run on the local RTX GPU.')
 ds=Windows().load(); assert len(ds), 'run --prepare first'; loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True,persistent_workers=args.workers>0); it=iter(loader)
 m=build_method(args.method).cuda(); opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=1e-4); amp=args.amp; scaler=torch.amp.GradScaler('cuda',enabled=amp)
 torch.cuda.reset_peak_memory_stats(); times=[]; loss=None
 for step in range(args.warmup_steps+args.steps):
  try: x=next(it)
  except StopIteration: it=iter(loader);x=next(it)
  x=x.cuda(non_blocking=True); torch.cuda.synchronize(); t=time.perf_counter(); opt.zero_grad(set_to_none=True)
  with torch.amp.autocast('cuda',enabled=amp): loss,_=m(x)
  scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
  if hasattr(m,'update_ema'):m.update_ema()
  torch.cuda.synchronize()
  if step>=args.warmup_steps: times.append(time.perf_counter()-t)
 out=dict(method=args.method,batch_size=args.batch_size,amp=amp,steps=args.steps,median_step_s=float(np.median(times)),mean_step_s=float(np.mean(times)),p90_step_s=float(np.percentile(times,90)),windows_s=args.batch_size/float(np.median(times)),peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30,loss=float(loss),gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,cases=len(ds.files),windows=len(ds))
 print(json.dumps(out,indent=2)); return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=200);p.add_argument('--method',choices=['mae','data2vec','jepa']);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--steps',type=int,default=200);p.add_argument('--warmup-steps',type=int,default=30);p.add_argument('--amp',action='store_true');p.add_argument('--workers',type=int,default=2);args=p.parse_args(); prepare(args.start,args.count) if args.prepare else train(args)
