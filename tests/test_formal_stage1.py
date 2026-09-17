import csv, numpy as np, torch
import pandas as pd
from scripts.prepare_vitaldb_ppg import prepare, VERSION
from src.models.common import BackboneConfig
from src.models.mae import MAE1D, masked_mse_loss, patch_targets
from src.data.processed_dataset import ProcessedPPGDataset
from src.models.mae.model import patch_targets
from src.models.common.patch_embed import PatchEmbed1D
from src.pretrain.optim import adamw_parameter_groups
def test_mae_shapes_mask_and_backward():
    m=MAE1D(BackboneConfig(16,2,4,2),patch_size=10,mask_ratio=.5,decoder_dim=8,decoder_depth=1,decoder_num_heads=2); o=m(torch.randn(2,1,50),generator=torch.Generator().manual_seed(3)); assert o.prediction.shape==(2,5,10) and o.mask.sum(1).tolist()==[3,3] and torch.isfinite(o.loss); o.loss.backward()
def test_masked_loss_ignores_unmasked_targets():
    p=torch.zeros(1,2,3); t=torch.ones_like(p); mask=torch.tensor([[1,0]],dtype=torch.bool); a=masked_mse_loss(p,t,mask); t[:,1]=99; assert masked_mse_loss(p,t,mask)==a
def test_dataset_mapping_and_provenance(tmp_path):
    np.save(tmp_path/'w.npy',np.arange(10000,dtype=np.float32).reshape(2,5000)); np.save(tmp_path/'i.npy',np.array([4,9],dtype=np.int64));
    with (tmp_path/'m.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['caseid','tid','waveform_path','accepted_index_path','status']); w.writeheader(); w.writerow(dict(caseid='7',tid='x',waveform_path='w.npy',accepted_index_path='i.npy',status='complete'))
    d=ProcessedPPGDataset(tmp_path/'m.csv'); assert len(d)==2 and d[1]['caseid']=='7' and d[1]['window_index']==9 and d[1]['waveform'].shape==(1,5000)
def test_dataset_version_and_length_contract(tmp_path):
    np.save(tmp_path/'w.npy',np.zeros((1,5000),np.float32)); np.save(tmp_path/'i.npy',np.array([2],np.int64));
    with (tmp_path/'m.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['caseid','tid','waveform_path','accepted_index_path','status','preprocessing_version']); w.writeheader(); w.writerow(dict(caseid='1',tid='t',waveform_path='w.npy',accepted_index_path='i.npy',status='complete',preprocessing_version='old'))
    try: ProcessedPPGDataset(tmp_path/'m.csv',expected_preprocessing_version='ppg_preprocessing_v0')
    except ValueError as e: assert 'caseid=1' in str(e) and 'old' in str(e)
    else: raise AssertionError('stale version accepted')
    try: ProcessedPPGDataset(tmp_path/'m.csv',expected_length=10)
    except ValueError: pass
    else: raise AssertionError('wrong expected length accepted')
def test_dataset_exposes_failed_rows_without_indexing_them(tmp_path):
    np.save(tmp_path/'w.npy',np.zeros((1,5000),np.float32)); np.save(tmp_path/'i.npy',np.array([3],np.int64))
    with (tmp_path/'m.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['caseid','tid','waveform_path','accepted_index_path','status']); w.writeheader(); w.writerow(dict(caseid='1',tid='ok',waveform_path='w.npy',accepted_index_path='i.npy',status='complete')); w.writerow(dict(caseid='2',tid='bad',waveform_path='missing.npy',accepted_index_path='missing_i.npy',status='failed'))
    d=ProcessedPPGDataset(tmp_path/'m.csv'); assert (d.total_manifest_rows,d.complete_count,d.failed_count,len(d))==(2,1,1,1) and d.failed_rows[0]['caseid']=='2'
def _source_csv(path,rows):
    with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['caseid','tid','local_path']); w.writeheader(); w.writerows(rows)
def test_prepare_rejects_duplicate_flat_path_and_stem_collision(tmp_path):
    raw=tmp_path/'raw'; raw.mkdir(); processed=tmp_path/'processed'; out=processed/'m.csv'
    dup=tmp_path/'dup.csv'; _source_csv(dup,[{'caseid':'1','tid':'a','local_path':'a.csv'},{'caseid':'1','tid':'a','local_path':'b.csv'}])
    try: prepare(raw,processed,dup,out)
    except ValueError as e: assert 'duplicate source key' in str(e)
    else: raise AssertionError('duplicate key accepted')
    nested=tmp_path/'nested.csv'; _source_csv(nested,[{'caseid':'1','tid':'a','local_path':'sub/a.csv'}])
    try: prepare(raw,processed,nested,out)
    except ValueError as e: assert 'caseid=1' in str(e) and 'sub/a.csv' in str(e)
    else: raise AssertionError('nested path accepted')
    collision=tmp_path/'collision.csv'; _source_csv(collision,[{'caseid':'1','tid':'a','local_path':'same.csv'},{'caseid':'2','tid':'b','local_path':'same.csv'}])
    try: prepare(raw,processed,collision,out)
    except ValueError as e: assert 'stem collision' in str(e)
    else: raise AssertionError('stem collision accepted')
def test_overlapping_targets_match_embed_and_mask_ratio_bounds():
    x=torch.randn(2,1,23); y=patch_targets(x,7,4); tok,_=PatchEmbed1D(1,8,7,4)(x); assert y.shape[1]==tok.shape[1]
    for ratio in (0.,1.):
        try: MAE1D(BackboneConfig(8,1,2,2),patch_size=4,mask_ratio=ratio,decoder_dim=4,decoder_num_heads=2)
        except ValueError: pass
        else: raise AssertionError('boundary ratio accepted')
def test_mask_generator_reproducible_and_optimizer_groups():
    m=MAE1D(BackboneConfig(8,1,2,2),patch_size=4,mask_ratio=.5,decoder_dim=4,decoder_num_heads=2); a=m(torch.randn(1,1,20),generator=torch.Generator().manual_seed(4)).mask; b=m(torch.randn(1,1,20),generator=torch.Generator().manual_seed(4)).mask; assert torch.equal(a,b)
    groups=adamw_parameter_groups(m,.1); decay={id(p) for p in groups[0]['params']}; no_decay={id(p) for p in groups[1]['params']}; all_train={id(p) for p in m.parameters() if p.requires_grad}; assert decay.isdisjoint(no_decay) and decay|no_decay==all_train and groups[0]['weight_decay']==.1 and groups[1]['weight_decay']==0
    for name,p in m.named_parameters():
        if name.endswith('bias') or 'norm' in name.lower() or p.ndim<=1: assert id(p) in no_decay
        if (name.endswith('weight') and p.ndim>=2 and 'norm' not in name.lower()) or name=='patch.proj.weight': assert id(p) in decay
    assert id(m.mask_token) in decay  # current explicit policy: ordinary rank-3 learned token is decayed
def test_prepare_interruption_persists_prior_source(tmp_path, monkeypatch):
    raw,source=_raw_fixture(tmp_path,count=2); processed=tmp_path/'processed'; manifest=processed/'manifest.csv'; import scripts.prepare_vitaldb_ppg as prep; original=prep.preprocess_ppg_record; calls=[]
    def interrupt_on_second(x,*a,**k):
        calls.append(k.get('caseid'))
        if str(k.get('caseid'))=='2': raise KeyboardInterrupt('deliberate')
        return original(x,*a,**k)
    monkeypatch.setattr(prep,'preprocess_ppg_record',interrupt_on_second)
    try: prepare(raw,processed,source,manifest)
    except KeyboardInterrupt: pass
    rows=list(csv.DictReader(manifest.open())); assert len(rows)==1 and rows[0]['caseid']=='1' and rows[0]['status']=='complete'
    calls.clear(); monkeypatch.setattr(prep,'preprocess_ppg_record',lambda x,*a,**k:(calls.append(k.get('caseid')) or original(x,*a,**k))); prepare(raw,processed,source,manifest); assert calls==['2']

def test_repeated_restart_preserves_unvisited_complete_rows(tmp_path, monkeypatch):
    raw,source=_raw_fixture(tmp_path,count=3); processed=tmp_path/'processed'; manifest=processed/'manifest.csv'; prepare(raw,processed,source,manifest)
    rows=list(csv.DictReader(manifest.open())); rows[1]['preprocessing_version']='stale'
    with manifest.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    import scripts.prepare_vitaldb_ppg as prep; original=prep.preprocess_ppg_record
    def interrupt_on_b(x,*a,**k):
        if str(k.get('caseid'))=='2': raise KeyboardInterrupt('deliberate later-source interruption')
        return original(x,*a,**k)
    monkeypatch.setattr(prep,'preprocess_ppg_record',interrupt_on_b)
    try: prepare(raw,processed,source,manifest)
    except KeyboardInterrupt: pass
    persisted=list(csv.DictReader(manifest.open())); assert [r['caseid'] for r in persisted]==['1','2','3']
    calls=[]; monkeypatch.setattr(prep,'preprocess_ppg_record',lambda x,*a,**k:(calls.append(k.get('caseid')) or original(x,*a,**k))); prepare(raw,processed,source,manifest); assert calls==['2']

def _raw_fixture(tmp_path, count=1):
    raw=tmp_path/'raw'; raw.mkdir(); rows=[]
    t=np.arange(5000*2)/500.; x=np.sin(2*np.pi*2*t)
    for n in range(count):
        name=f'case_{n+1:04d}__tid{n+1}.csv'; pd.DataFrame({'SNUADC/PLETH':x}).to_csv(raw/name,index=False); rows.append({'caseid':str(n+1),'tid':f'tid{n+1}','local_path':name})
    source=tmp_path/'sources.csv'
    with source.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['caseid','tid','local_path']); w.writeheader(); w.writerows(rows)
    return raw,source

def test_prepare_dataset_end_to_end_and_portable_paths(tmp_path):
    raw,source=_raw_fixture(tmp_path); processed=tmp_path/'processed'; manifest=processed/'manifest.csv'; prepare(raw,processed,source,manifest)
    row=next(csv.DictReader(manifest.open())); assert not str(processed) in row['waveform_path'] and row['waveform_path']=='case_0001__tid1.npy'
    d=ProcessedPPGDataset(manifest,processed); item=d[0]; assert item['waveform'].shape==(1,5000) and item['caseid']=='1' and item['window_index']==0
    raw.joinpath('case_0001__tid1.csv').unlink(); assert d[1]['window_index']==1

def test_prepare_resume_version_and_partial_outputs(tmp_path, monkeypatch):
    raw,source=_raw_fixture(tmp_path); processed=tmp_path/'processed'; manifest=processed/'manifest.csv'; prepare(raw,processed,source,manifest)
    import scripts.prepare_vitaldb_ppg as prep; original=prep.preprocess_ppg_record; calls=[]
    monkeypatch.setattr(prep,'preprocess_ppg_record',lambda *a,**k:(calls.append(1) or original(*a,**k))); prepare(raw,processed,source,manifest); assert calls==[]
    rows=list(csv.DictReader(manifest.open())); rows[0]['preprocessing_version']='stale'
    with manifest.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    prepare(raw,processed,source,manifest); assert calls and next(csv.DictReader(manifest.open()))['preprocessing_version']==VERSION
    processed.joinpath('case_0001__tid1__accepted_indices.npy').unlink(); calls.clear(); prepare(raw,processed,source,manifest); assert calls

import pytest
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for AMP regression')
def test_mae_cuda_amp_forward_backward():
    m=MAE1D(BackboneConfig(16,2,4,2),patch_size=10,mask_ratio=.5,decoder_dim=8,decoder_depth=1,decoder_num_heads=2).cuda(); x=torch.randn(2,1,50,device='cuda')
    with torch.autocast(device_type='cuda',dtype=torch.float16): o=m(x,generator=torch.Generator(device='cpu').manual_seed(3))
    assert torch.isfinite(o.loss); o.loss.backward(); assert m.mask_token.grad is not None and torch.isfinite(m.mask_token.grad).all()
