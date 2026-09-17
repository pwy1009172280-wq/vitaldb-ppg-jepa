from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class ModelConfig:
    patch_size:int=50; patch_stride:int=50; embed_dim:int=128; depth:int=4; num_heads:int=4; mlp_ratio:float=4.; dropout:float=0.; decoder_dim:int=64; decoder_depth:int=2; decoder_num_heads:int=4; decoder_mlp_ratio:float=4.
    def validate(self):
        for n in ('patch_size','patch_stride','embed_dim','depth','num_heads','decoder_dim','decoder_depth','decoder_num_heads'):
            if getattr(self,n)<=0: raise ValueError(f'{n} must be > 0')
        if self.embed_dim%2 or self.embed_dim%self.num_heads: raise ValueError('embed_dim must be even and divisible by num_heads')
        if self.decoder_dim%2 or self.decoder_dim%self.decoder_num_heads: raise ValueError('decoder_dim must be even and divisible by decoder_num_heads')
        if self.mlp_ratio<=0 or self.decoder_mlp_ratio<=0 or not 0<=self.dropout<1: raise ValueError('invalid ratios/dropout')
        return self
@dataclass(frozen=True)
class DataConfig: manifest:str='data/processed/manifest.csv'; processed_root:str='data/processed'; input_length:int=5000; expected_preprocessing_version:str='ppg_preprocessing_v0'
@dataclass(frozen=True)
class TrainConfig: batch_size:int=16; epochs:int=1; max_updates:int|None=None; learning_rate:float=1e-4; weight_decay:float=.05; warmup_updates:int=0; min_lr_ratio:float=0.; seed:int=0; amp:bool=True; num_workers:int=0; pin_memory:bool=True; drop_last:bool=True; grad_accum_steps:int=1; checkpoint_interval_updates:int=100; log_interval_updates:int=10; grad_clip_norm:float|None=None; run_dir:str='runs/mae'; device:str='auto'; allow_incomplete:bool=False
@dataclass(frozen=True)
class MAEConfig: mask_ratio:float=.75
@dataclass(frozen=True)
class Data2VecConfig: mask_ratio:float=.65; mask_span_length:int=4; target_top_k_layers:int=2; ema_momentum:float=.996; loss_beta:float=1.
@dataclass(frozen=True)
class JEPAConfig: num_target_blocks:int=2; target_block_length:int=4; predictor_dim:int=128; predictor_depth:int=2; predictor_num_heads:int=4; predictor_mlp_ratio:float=4.; ema_momentum:float=.996; loss_beta:float=1.
@dataclass(frozen=True)
class SSLConfig:
    method:str='mae'; model:ModelConfig=field(default_factory=ModelConfig); data:DataConfig=field(default_factory=DataConfig); train:TrainConfig=field(default_factory=TrainConfig); mae:MAEConfig=field(default_factory=MAEConfig); data2vec:Data2VecConfig=field(default_factory=Data2VecConfig); jepa:JEPAConfig=field(default_factory=JEPAConfig)
    def validate(self):
        if self.method not in ('mae','data2vec','jepa'): raise ValueError('supported methods are mae, data2vec, and jepa')
        self.model.validate()
        if self.data.input_length<self.model.patch_size or self.train.batch_size<=0 or self.train.epochs<=0 or self.train.learning_rate<=0 or self.train.weight_decay<0 or self.train.warmup_updates<0 or not 0<=self.train.min_lr_ratio<=1 or self.train.grad_accum_steps!=1 or self.train.log_interval_updates<=0 or self.train.checkpoint_interval_updates<=0: raise ValueError('invalid data/training values; intervals must be > 0 and grad_accum_steps must equal 1')
        if self.method=='mae' and not 0<self.mae.mask_ratio<1: raise ValueError('mask_ratio must be in (0,1)')
        if self.method=='data2vec' and (not 0<self.data2vec.mask_ratio<1 or self.data2vec.mask_span_length<=0 or not 1<=self.data2vec.target_top_k_layers<=self.model.depth or not 0<=self.data2vec.ema_momentum<=1 or self.data2vec.loss_beta<0): raise ValueError('invalid data2vec values')
        if self.method=='jepa' and (self.jepa.num_target_blocks<=0 or self.jepa.target_block_length<=0 or self.jepa.predictor_dim<=0 or self.jepa.predictor_num_heads<=0 or self.jepa.predictor_dim%2 or self.jepa.predictor_dim%self.jepa.predictor_num_heads or self.jepa.predictor_depth<=0 or self.jepa.predictor_mlp_ratio<=0 or not 0<=self.jepa.ema_momentum<=1 or self.jepa.loss_beta<0 or self.model.patch_stride!=self.model.patch_size): raise ValueError('invalid JEPA values')
        return self
def load_config(path:str|Path, overrides:dict[str,Any]|None=None):
    raw=yaml.safe_load(Path(path).read_text()) or {}
    if overrides: raw.update(overrides)
    get=lambda cls,k: cls(**(raw.get(k) or {}))
    return SSLConfig(raw.get('method','mae'),get(ModelConfig,'model'),get(DataConfig,'data'),get(TrainConfig,'train'),get(MAEConfig,'mae'),get(Data2VecConfig,'data2vec'),get(JEPAConfig,'jepa')).validate()
