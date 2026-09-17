from dataclasses import dataclass
import copy,torch
from torch import nn
import torch.nn.functional as F
from ..common import BackboneConfig,PatchEmbed1D,TransformerEncoderBackbone
from ..common.masking import select_tokens
from .masking import make_target_block_masks
@dataclass
class JEPAOutput:
    loss:torch.Tensor; target_mask:torch.Tensor; context_mask:torch.Tensor; prediction:torch.Tensor|None; target:torch.Tensor|None; target_count:int; target_fraction:float; target_mean:torch.Tensor; target_std:torch.Tensor
class JEPA1D(nn.Module):
    def __init__(self,config:BackboneConfig,patch_size=50,patch_stride=None,num_target_blocks=2,target_block_length=4,predictor_dim=None,predictor_depth=2,predictor_num_heads=4,predictor_mlp_ratio=4.,ema_momentum=.996,loss_beta=1.):
        super().__init__(); stride=patch_stride or patch_size
        if stride!=patch_size: raise ValueError('vanilla JEPA requires patch_stride == patch_size')
        if predictor_dim is not None and predictor_dim<=0: raise ValueError('JEPA predictor_dim must be > 0')
        if predictor_num_heads<=0: raise ValueError('JEPA predictor_num_heads must be > 0')
        if predictor_mlp_ratio<=0: raise ValueError('JEPA predictor_mlp_ratio must be > 0')
        pd=config.embed_dim if predictor_dim is None else predictor_dim
        if num_target_blocks<=0 or target_block_length<=0 or predictor_depth<=0 or pd%2 or pd%predictor_num_heads or not 0<=ema_momentum<=1 or loss_beta<0: raise ValueError('invalid JEPA configuration')
        self.patch_size=patch_size; self.patch_stride=stride; self.num_target_blocks=num_target_blocks; self.target_block_length=target_block_length; self.predictor_dim=pd; self.ema_momentum=ema_momentum; self.loss_beta=loss_beta
        self.online_patch=PatchEmbed1D(1,config.embed_dim,patch_size,stride); self.context_encoder=TransformerEncoderBackbone(config); self.context_projection=nn.Linear(config.embed_dim,pd); self.target_query=nn.Parameter(torch.zeros(1,1,pd)); self.predictor=TransformerEncoderBackbone(BackboneConfig(pd,predictor_depth,predictor_num_heads,predictor_mlp_ratio,config.dropout,'sinusoidal')); self.predictor_projection=nn.Linear(pd,config.embed_dim)
        self.target_patch=copy.deepcopy(self.online_patch); self.target_encoder=copy.deepcopy(self.context_encoder)
        for p in list(self.target_patch.parameters())+list(self.target_encoder.parameters()): p.requires_grad_(False)
        self.target_patch.eval(); self.target_encoder.eval()
    def train(self,mode=True): super().train(mode); self.target_patch.eval(); self.target_encoder.eval(); return self
    def _target_features(self,x):
        with torch.no_grad(): t,p=self.target_patch(x); return self.target_encoder(t,p).tokens
    def forward(self,x,*,generator=None,return_prediction=True,return_target=True):
        t,p=self.online_patch(x); b,n,d=t.shape; masks=make_target_block_masks(b,n,self.num_target_blocks,self.target_block_length,generator=generator,device=x.device); c,cp=select_tokens(t,p,masks.context_mask); c=self.context_encoder(c,cp).tokens; projected_context=self.context_projection(c); full=self.target_query.to(dtype=projected_context.dtype).expand(b,n,-1).clone(); full[masks.context_mask]=projected_context.reshape(-1,projected_context.shape[-1]); pred=self.predictor(full,p).tokens; pred,_=select_tokens(pred,p,masks.target_mask); target=self._target_features(x); target,_=select_tokens(target,p,masks.target_mask); out=self.predictor_projection(pred); values=out-target; loss=F.smooth_l1_loss(values,torch.zeros_like(values),beta=self.loss_beta,reduction='mean'); return JEPAOutput(loss,masks.target_mask,masks.context_mask,out if return_prediction else None,target if return_target else None,int(masks.target_mask.sum(1)[0]),float(masks.target_mask.float().mean()),target.mean().detach(),target.std(unbiased=False).detach())
    @torch.no_grad()
    def ema_update(self,momentum=None):
        m=self.ema_momentum if momentum is None else momentum
        if not 0<=m<=1: raise ValueError('momentum must be in [0,1]')
        for target,online in ((self.target_patch,self.online_patch),(self.target_encoder,self.context_encoder)):
            for tp,op in zip(target.parameters(),online.parameters()): tp.mul_(m).add_(op,alpha=1-m)
