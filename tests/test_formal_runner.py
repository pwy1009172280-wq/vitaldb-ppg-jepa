import csv, copy, tarfile
from pathlib import Path
import numpy as np, yaml, torch
from scripts.train_ssl import run_training, planned_updates, resume_compatibility_errors, main
from scripts.package_for_cluster import package, manifest

def _fixture(tmp_path):
 root=tmp_path/'processed'; root.mkdir(); np.save(root/'x.npy',np.ones((4,5000),np.float32)); np.save(root/'x__accepted_indices.npy',np.arange(4,dtype=np.int64)); m=root/'manifest.csv'
 fields=['caseid','tid','waveform_path','accepted_index_path','source_windows','accepted_windows','rejected_windows','preprocessing_version','status','error']
 with m.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerow(dict(caseid='1',tid='x',waveform_path='x.npy',accepted_index_path='x__accepted_indices.npy',source_windows='4',accepted_windows='4',rejected_windows='0',preprocessing_version='ppg_preprocessing_v0',status='complete',error=''))
 return root,m

def _cfg(tmp,method,root,m,run,max_updates=3):
 c={'method':method,'model':{'patch_size':50,'patch_stride':50,'embed_dim':8,'depth':1,'num_heads':2,'mlp_ratio':2.,'dropout':0.,'decoder_dim':4,'decoder_depth':1,'decoder_num_heads':2,'decoder_mlp_ratio':2.},'data':{'manifest':str(m),'processed_root':str(root),'input_length':5000,'expected_preprocessing_version':'ppg_preprocessing_v0'},'train':{'batch_size':2,'epochs':2,'max_updates':max_updates,'learning_rate':1e-3,'weight_decay':.01,'warmup_updates':0,'min_lr_ratio':0.,'seed':4,'amp':False,'num_workers':0,'pin_memory':False,'drop_last':True,'grad_accum_steps':1,'checkpoint_interval_updates':1,'log_interval_updates':1,'run_dir':str(run),'device':'cpu','allow_incomplete':False}}
 c[method]={'mask_ratio':.5} if method=='mae' else ({'mask_ratio':.5,'mask_span_length':1,'target_top_k_layers':1,'ema_momentum':.9,'loss_beta':1.} if method=='data2vec' else {'num_target_blocks':1,'target_block_length':1,'predictor_dim':8,'predictor_depth':1,'predictor_num_heads':2,'predictor_mlp_ratio':2.,'ema_momentum':.9,'loss_beta':1.})
 return c

def test_true_resume_matches_reference_all_methods(tmp_path):
 root,m=_fixture(tmp_path)
 for method in ('mae','data2vec','jepa'):
  ref=tmp_path/f'{method}_ref'; ir=tmp_path/f'{method}_int'; cp=tmp_path/f'{method}.yaml'; c=_cfg(tmp_path,method,root,m,ref); cp.write_text(yaml.safe_dump(c)); a=run_training(__import__('src.pretrain.config',fromlist=['load_config']).load_config(cp)); c['train']['run_dir']=str(ir); cp.write_text(yaml.safe_dump(c)); b=run_training(__import__('src.pretrain.config',fromlist=['load_config']).load_config(cp),stop_after_successful_updates=1); run_training(__import__('src.pretrain.config',fromlist=['load_config']).load_config(cp),resume=str(b['checkpoint'])); ca=torch.load(ref/'checkpoints/last.pt',weights_only=False); cb=torch.load(ir/'checkpoints/last.pt',weights_only=False); assert ca['global_update']==cb['global_update']==3; assert ca['scheduler']==cb['scheduler']; assert ca['rng']['mask'].equal(cb['rng']['mask']); assert all(torch.allclose(ca['model'][k],cb['model'][k]) for k in ca['model'])

def test_planned_update_rules():
 assert planned_updates(5,2,3,True,None)==6; assert planned_updates(5,2,3,False,None)==9; assert planned_updates(5,2,3,False,4)==4

def test_resume_mismatch_and_operational_fields():
 saved={'method':'mae','model':dict(patch_size=1,patch_stride=1,embed_dim=2,depth=1,num_heads=1,mlp_ratio=2.,dropout=0.),'data':{'input_length':5000,'expected_preprocessing_version':'v'},'train':dict(batch_size=1,learning_rate=1e-3,weight_decay=0.,warmup_updates=0,min_lr_ratio=0.,seed=1,amp=False,drop_last=True,grad_clip_norm=None,grad_accum_steps=1,epochs=1,max_updates=1,allow_incomplete=False),'mae':{'mask_ratio':.5}}; cur=copy.deepcopy(saved); cur['train']['learning_rate']=2e-3; assert any('train.learning_rate' in x for x in resume_compatibility_errors(saved,cur)); cur['train']['learning_rate']=1e-3; cur['train']['num_workers']=4; assert not resume_compatibility_errors(saved,cur)

def test_package_allowlist_and_manifest(tmp_path):
 for rel in ('src/data/__init__.py','src/data/ppg_preprocessing.py','src/data/processed_dataset.py','src/models/common/__init__.py','src/models/common/config.py','src/models/common/masking.py','src/models/common/patch_embed.py','src/models/common/positional_encoding.py','src/models/common/transformer.py','src/models/common/types.py','src/models/mae/__init__.py','src/models/mae/model.py','scripts/train_ssl.py','configs/smoke/mae.yaml','docs/cluster_runbook.md','tests/test_formal_runner.py','README.md','.env','notes.txt','runs/run1/checkpoint.pt','src/models/ssl_benchmarks.py'):
  p=tmp_path/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('x')
 out=tmp_path/'x.tar.gz'; package(tmp_path,out); names=[]
 with tarfile.open(out) as t:names=t.getnames()
 assert 'src/models/mae/model.py' in names and 'PACKAGE_MANIFEST.txt' in names
 assert '.env' not in names and 'notes.txt' not in names and 'src/models/ssl_benchmarks.py' not in names

def test_intervals_gate_normal_events_but_force_stop_checkpoint(tmp_path, monkeypatch):
    root,m=_fixture(tmp_path); c=_cfg(tmp_path,'mae',root,m,tmp_path/'interval',max_updates=2)
    c['train']['log_interval_updates']=2; c['train']['checkpoint_interval_updates']=2
    p=tmp_path/'i.yaml'; p.write_text(yaml.safe_dump(c)); from src.pretrain.config import load_config; import scripts.train_ssl as runner
    calls=[]; real=runner._checkpoint
    monkeypatch.setattr(runner,'_checkpoint',lambda *args,**kw:(calls.append(args[6]),real(*args,**kw)))
    run_training(load_config(p),stop_after_successful_updates=1)
    assert calls == [1]
    assert not (tmp_path/'interval'/'metrics.jsonl').exists()

def test_cli_path_overrides_are_applied(monkeypatch):
    import scripts.train_ssl as runner
    captured=[]
    monkeypatch.setattr(runner, 'run_training', lambda cfg, *args, **kwargs: captured.append(cfg) or {})
    monkeypatch.setattr('sys.argv', ['train_ssl.py', '--config', 'configs/smoke/mae.yaml',
                                     '--manifest', '/cluster/manifest.csv',
                                     '--processed-root', '/cluster/processed',
                                     '--run-dir', '/cluster/runs/mae', '--device', 'cpu'])
    assert main() == 0
    cfg=captured[0]
    assert cfg.data.manifest == '/cluster/manifest.csv'
    assert cfg.data.processed_root == '/cluster/processed'
    assert cfg.train.run_dir == '/cluster/runs/mae'
    assert cfg.train.device == 'cpu'
    assert cfg.data.input_length == 5000
    assert cfg.data.expected_preprocessing_version == 'ppg_preprocessing_v0'
