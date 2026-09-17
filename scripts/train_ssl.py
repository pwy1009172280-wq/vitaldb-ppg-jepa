#!/usr/bin/env python3
import argparse,datetime,signal,sys,time
from pathlib import Path
import torch,yaml
from torch.utils.data import DataLoader,Subset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pretrain.config import load_config,SSLConfig
from src.pretrain.runner_utils import *
from src.data.processed_dataset import ProcessedPPGDataset
from src.models.factory import build_model
from src.pretrain.optim import adamw_parameter_groups

def effective_config_dict(cfg):
 return {'method':cfg.method,'model':dict(cfg.model.__dict__),'data':dict(cfg.data.__dict__),'train':dict(cfg.train.__dict__),cfg.method:dict(getattr(cfg,cfg.method).__dict__)}

def resume_compatibility_errors(saved,current):
 paths=[('method',),('model','patch_size'),('model','patch_stride'),('model','embed_dim'),('model','depth'),('model','num_heads'),('model','mlp_ratio'),('model','dropout'),('data','input_length'),('data','expected_preprocessing_version')]
 paths += [('train',x) for x in ('batch_size','learning_rate','weight_decay','warmup_updates','min_lr_ratio','seed','amp','drop_last','grad_clip_norm','grad_accum_steps','epochs','max_updates','allow_incomplete')]
 if saved.get('method') in current: paths += [(saved['method'],k) for k in current[saved['method']]]
 errors=[]
 for path in paths:
  a=saved; b=current; label='.'.join(path)
  try:
   for k in path:a=a[k]
  except (KeyError,TypeError): errors.append(f'- {label}: missing in checkpoint'); continue
  try:
   for k in path:b=b[k]
  except (KeyError,TypeError): errors.append(f'- {label}: missing in current resolved config'); continue
  if a!=b: errors.append(f"- {label}: checkpoint={a!r} current={b!r}")
 return errors

def planned_updates(n,batch,epochs,drop_last,max_updates):
 batches=n//batch if drop_last else (n+batch-1)//batch
 if batches==0: raise ValueError('batch_size and drop_last produce zero batches per epoch')
 budget=epochs*batches
 if max_updates is not None and max_updates<=0: raise ValueError('max_updates must be > 0')
 return min(max_updates,budget) if max_updates is not None else budget

def _checkpoint(run,model,opt,sched,scaler,epoch,next_batch,update,cfg,mask):
 obj={'format_version':1,'method':cfg.method,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'next_batch_idx':next_batch,'global_update':update,'resolved_config':effective_config_dict(cfg),'rng':capture_rng(mask),'git_commit':git_commit(Path(__file__).resolve().parents[1]),'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 atomic_torch_save(obj,run/'checkpoints'/'last.pt')

def run_training(cfg,resume=None,stop_after_successful_updates=None):
 cfg.validate(); data,train=cfg.data,cfg.train
 device=torch.device('cuda' if train.device=='auto' and torch.cuda.is_available() else ('cpu' if train.device=='auto' else train.device))
 ds=ProcessedPPGDataset(data.manifest,data.processed_root,data.expected_preprocessing_version,data.input_length)
 if not len(ds): raise ValueError('processed dataset is empty')
 if ds.failed_count and not train.allow_incomplete: raise ValueError(f'processed manifest has {ds.failed_count} failed rows')
 run=Path(train.run_dir); run.mkdir(parents=True,exist_ok=True); (run/'checkpoints').mkdir(exist_ok=True); (run/'train.log').touch(); resolved=run/'resolved_config.yaml'; current=effective_config_dict(cfg)
 if resolved.exists() and resume is None: raise FileExistsError(f'run already exists: {run}')
 if resume is None: resolved.write_text(yaml.safe_dump(current,sort_keys=False))
 seed_everything(train.seed); model=build_model(cfg).to(device)
 opt=torch.optim.AdamW(adamw_parameter_groups(model,train.weight_decay),lr=train.learning_rate)
 total=planned_updates(len(ds),train.batch_size,train.epochs,train.drop_last,train.max_updates)
 sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda u:cosine_lambda(u,train.warmup_updates,total,train.min_lr_ratio)); amp=bool(train.amp and device.type=='cuda'); scaler=torch.amp.GradScaler('cuda',enabled=amp); mask=torch.Generator(device='cpu').manual_seed(train.seed+7919); epoch=next_batch=update=0
 if resume:
  ck=torch.load(resume,map_location='cpu',weights_only=False)
  if ck.get('format_version') != 1: raise ValueError(f"unsupported checkpoint format_version: {ck.get('format_version')!r}")
  errors=resume_compatibility_errors(ck.get('resolved_config',{}),current)
  if errors: raise ValueError('resume configuration mismatch:\n'+'\n'.join(errors))
  model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck.get('scaler',{})); epoch,next_batch,update=ck['epoch'],ck['next_batch_idx'],ck['global_update']; restore_rng(ck['rng'],mask)
 stop=[False]; signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True)); signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True)); metrics=run/'metrics.jsonl'; session_start_update=update; start=time.time()
 while epoch<train.epochs and update<total:
  perm=epoch_permutation(len(ds),train.seed,epoch); lg=torch.Generator(device='cpu').manual_seed(train.seed+104729+epoch)
  loader=DataLoader(Subset(ds,perm),batch_size=train.batch_size,shuffle=False,generator=lg,num_workers=train.num_workers,pin_memory=train.pin_memory,drop_last=train.drop_last,worker_init_fn=worker_init_fn)
  for bi,b in enumerate(loader):
   if bi<next_batch: continue
   if update>=total: break
   x=b['waveform'].to(device,non_blocking=True); opt.zero_grad(set_to_none=True); before=scaler.get_scale()
   with torch.autocast(device_type='cuda',enabled=amp): out=model(x,generator=mask); loss=out.loss
   if not torch.isfinite(loss): raise FloatingPointError('non-finite loss')
   if amp: scaler.scale(loss).backward(); scaler.unscale_(opt)
   else: loss.backward()
   if train.grad_clip_norm is not None: torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],train.grad_clip_norm)
   if amp: scaler.step(opt); scaler.update(); ok=scaler.get_scale()>=before
   else: opt.step(); ok=True
   if ok:
    update+=1; sched.step()
    if hasattr(model,'ema_update'): model.ema_update()
    event={'method':cfg.method,'epoch':epoch,'batch_idx':bi,'global_update':update,'loss':float(loss.detach()),'lr':opt.param_groups[0]['lr'],'elapsed_s':time.time()-start,'device':str(device),'peak_gpu_memory':torch.cuda.max_memory_allocated() if device.type=='cuda' else None}
    for key in ('masked_fraction','target_fraction','target_mean','target_std'):
     if hasattr(out,key): event[key]=float(getattr(out,key))
    if update % train.log_interval_updates == 0:
     event['session_elapsed_s']=time.time()-start; event['session_throughput_windows_s']=(update-session_start_update)*train.batch_size/max(event['session_elapsed_s'],1e-9); write_jsonl(metrics,event)
    if update % train.checkpoint_interval_updates == 0: _checkpoint(run,model,opt,sched,scaler,epoch,bi+1,update,cfg,mask)
    if stop_after_successful_updates is not None and update>=stop_after_successful_updates:
     _checkpoint(run,model,opt,sched,scaler,epoch,bi+1,update,cfg,mask); return {'global_update':update,'checkpoint':run/'checkpoints'/'last.pt'}
   if stop[0]: _checkpoint(run,model,opt,sched,scaler,epoch,bi+1,update,cfg,mask); return {'global_update':update,'checkpoint':run/'checkpoints'/'last.pt'}
  epoch+=1; next_batch=0; _checkpoint(run,model,opt,sched,scaler,epoch,0,update,cfg,mask)
 return {'global_update':update,'checkpoint':run/'checkpoints'/'last.pt','model':model,'scheduler':sched,'mask_state':mask.get_state()}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--manifest'); p.add_argument('--processed-root'); p.add_argument('--run-dir'); p.add_argument('--resume'); p.add_argument('--device'); p.add_argument('--allow-incomplete',action='store_true'); a=p.parse_args(); cfg=load_config(a.config); data,train=cfg.data,cfg.train
 if a.manifest:data=type(data)(a.manifest,data.processed_root,data.input_length,data.expected_preprocessing_version)
 if a.processed_root:data=type(data)(data.manifest,a.processed_root,data.input_length,data.expected_preprocessing_version)
 d=dict(train.__dict__)
 if a.run_dir:d['run_dir']=a.run_dir
 if a.device:d['device']=a.device
 if a.allow_incomplete:d['allow_incomplete']=True
 train=type(train)(**d)
 return 0 if run_training(SSLConfig(cfg.method,cfg.model,data,train,cfg.mae,cfg.data2vec,cfg.jepa),a.resume) else 0
if __name__=='__main__':raise SystemExit(main())
