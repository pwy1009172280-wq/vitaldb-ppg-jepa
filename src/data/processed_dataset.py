import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
class ProcessedPPGDataset(Dataset):
    def __init__(self,manifest,processed_root=None,expected_preprocessing_version=None,expected_length=5000):
        self.manifest=Path(manifest); self.root=Path(processed_root) if processed_root else self.manifest.parent
        self.expected_preprocessing_version=expected_preprocessing_version; self.expected_length=expected_length
        if expected_length<=0: raise ValueError('expected_length must be > 0')
        if not self.root.exists() or not self.root.is_dir(): raise FileNotFoundError(f'processed_root does not exist: {self.root}')
        with self.manifest.open(newline='') as f: rows=list(csv.DictReader(f))
        self.total_manifest_rows=len(rows); self.failed_rows=[r for r in rows if r.get('status','complete')!='complete']; self.failed_count=len(self.failed_rows); self.complete_count=self.total_manifest_rows-self.failed_count
        self.rows=[r for r in rows if r.get('status','complete')=='complete']; self.arrays=[]; self.indices=[]; self.prefix=[0]
        for r in self.rows:
            if expected_preprocessing_version is not None:
                actual=r.get('preprocessing_version','')
                if not actual or actual!=expected_preprocessing_version: raise ValueError(f"preprocessing version mismatch for caseid={r.get('caseid')} tid={r.get('tid')}: actual={actual!r}, expected={expected_preprocessing_version!r}")
            wp,ip=self._path(r['waveform_path']),self._path(r['accepted_index_path'])
            if not wp.is_file(): raise FileNotFoundError(f'missing waveform file: {wp}')
            if not ip.is_file(): raise FileNotFoundError(f'missing accepted-index file: {ip}')
            a=np.load(wp,mmap_mode='r'); i=np.load(ip,mmap_mode='r')
            if a.ndim!=2 or a.shape[1]!=expected_length or len(a)!=len(i): raise ValueError(f'invalid processed row for caseid={r.get("caseid")} tid={r.get("tid")}: expected length {expected_length}')
            self.arrays.append(a); self.indices.append(i); self.prefix.append(self.prefix[-1]+len(i))
    def _path(self,p):
        p=Path(p); return p if p.is_absolute() else self.root/p
    def __len__(self): return self.prefix[-1]
    def __getitem__(self,index):
        if index<0:index+=len(self)
        if index<0 or index>=len(self): raise IndexError(index)
        c=int(np.searchsorted(self.prefix,index,side='right')-1); local=index-self.prefix[c]; r=self.rows[c]
        sample=np.array(self.arrays[c][local],dtype=np.float32,copy=True)
        return {'waveform':torch.from_numpy(sample).unsqueeze(0),'caseid':r.get('caseid'),'tid':r.get('tid'),'window_index':int(self.indices[c][local])}
