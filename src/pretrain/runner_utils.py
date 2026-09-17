import json,math,os,random,subprocess
from pathlib import Path
import numpy as np,torch
def seed_everything(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def worker_init_fn(worker_id):
 s=torch.initial_seed()%(2**32); random.seed(s); np.random.seed(s)
def epoch_permutation(n,seed,epoch): return torch.randperm(n,generator=torch.Generator(device='cpu').manual_seed(seed+epoch)).tolist()
def cosine_lambda(update,warmup,total,min_ratio):
 if update<warmup:return update/max(1,warmup)
 p=min(1.,(update-warmup)/max(1,total-warmup)); return min_ratio+(1-min_ratio)*(.5*(1+math.cos(math.pi*p)))
def atomic_torch_save(obj,path):
 path=Path(path); tmp=path.with_name(path.name+'.tmp')
 try:
  with tmp.open('wb') as f:
   torch.save(obj,f); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path)
 except Exception:
  try: tmp.unlink()
  except FileNotFoundError: pass
  raise
def git_commit(repo_root):
 try:
  return subprocess.run(['git','rev-parse','HEAD'],cwd=str(repo_root),check=True,capture_output=True,text=True).stdout.strip() or None
 except Exception: return None
def capture_rng(mask): return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'mask':mask.get_state()}
def restore_rng(s,mask):
 random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
 if torch.cuda.is_available() and s.get('cuda') is not None: torch.cuda.set_rng_state_all(s['cuda'])
 mask.set_state(s['mask'])
def write_jsonl(path,obj):
 with Path(path).open('a') as f: f.write(json.dumps(obj,default=str)+'\n'); f.flush(); os.fsync(f.fileno())
