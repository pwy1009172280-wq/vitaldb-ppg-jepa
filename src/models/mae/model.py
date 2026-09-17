from dataclasses import dataclass
import torch
from torch import nn
from ..common import BackboneConfig,PatchEmbed1D,TransformerEncoderBackbone
from ..common.masking import make_temporal_mask,select_tokens
@dataclass
class MAEOutput:
    loss:torch.Tensor; prediction:torch.Tensor|None; mask:torch.Tensor; masked_indices:torch.Tensor; num_patches:int
def patch_targets(x,patch_size,patch_stride):
    if x.ndim!=3 or x.shape[1]!=1: raise ValueError('input must be [B,1,T]')
    return x.unfold(-1,patch_size,patch_stride).squeeze(1)
def masked_mse_loss(prediction,target,mask):
    if prediction.shape!=target.shape or mask.shape!=prediction.shape[:2]: raise ValueError('shape mismatch')
    v=(prediction-target)[mask]
    return v.square().sum()/v.numel() if v.numel() else prediction.sum()*0.
class MAE1D(nn.Module):
    def __init__(self,config:BackboneConfig,patch_size=50,patch_stride=None,mask_ratio=.75,decoder_dim=64,decoder_depth=2,decoder_num_heads=4,decoder_mlp_ratio=4.):
        super().__init__(); self.patch_size=patch_size; self.patch_stride=patch_stride or patch_size; self.mask_ratio=mask_ratio
        if not 0<mask_ratio<1: raise ValueError('mask_ratio must be in (0,1)')
        self.patch=PatchEmbed1D(1,config.embed_dim,patch_size,self.patch_stride); self.encoder=TransformerEncoderBackbone(config); self.to_decoder=nn.Linear(config.embed_dim,decoder_dim); self.mask_token=nn.Parameter(torch.zeros(1,1,decoder_dim)); self.decoder=TransformerEncoderBackbone(BackboneConfig(decoder_dim,decoder_depth,decoder_num_heads,decoder_mlp_ratio,config.dropout,'sinusoidal')); self.head=nn.Linear(decoder_dim,patch_size)
    def forward(self,x,*,generator=None,return_prediction=True):
        tok,pos=self.patch(x); b,n,_=tok.shape
        mask=make_temporal_mask(b,n,self.mask_ratio,generator=generator,device=x.device); counts=mask.sum(dim=1)
        if not torch.all(counts==counts[0]): raise ValueError('MAE mask sampler returned unequal per-sample counts')
        k=int(counts[0])
        if not 0<k<n: raise ValueError(f'invalid MAE mask count: mask_ratio={self.mask_ratio}, num_tokens={n}, effective_masked_count={k}')
        vis,vpos=select_tokens(tok,pos,~mask); enc=self.encoder(vis,vpos).tokens; visible_dec=self.to_decoder(enc); full=self.mask_token.to(dtype=visible_dec.dtype).expand(b,n,-1).clone(); full[~mask]=visible_dec.reshape(-1,visible_dec.shape[-1]); dec=self.decoder(full,pos).tokens; pred=self.head(dec); loss=masked_mse_loss(pred,patch_targets(x,self.patch_size,self.patch_stride),mask); idx=mask.nonzero().view(b,k,2)[...,1]; return MAEOutput(loss,pred if return_prediction else None,mask,idx,n)
