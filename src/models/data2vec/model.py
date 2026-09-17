from dataclasses import dataclass
import copy,torch
from torch import nn
import torch.nn.functional as F
from ..common import BackboneConfig,PatchEmbed1D,TransformerEncoderBackbone
from .masking import make_temporal_span_mask
@dataclass
class Data2VecOutput:
    loss:torch.Tensor; mask:torch.Tensor; prediction:torch.Tensor|None; target:torch.Tensor|None; masked_fraction:float; target_mean:torch.Tensor; target_std:torch.Tensor
class Data2VecStyle1D(nn.Module):
    """Documented data2vec-style PPG adaptation, not exact Fairseq reproduction."""
    def __init__(self,config:BackboneConfig,patch_size=50,patch_stride=None,mask_ratio=.65,mask_span_length=4,target_top_k_layers=2,ema_momentum=.996,loss_beta=1.):
        super().__init__(); stride=patch_stride or patch_size
        if not 0<mask_ratio<1 or mask_span_length<=0 or not 1<=target_top_k_layers<=config.depth or not 0<=ema_momentum<=1 or loss_beta<0: raise ValueError('invalid data2vec configuration')
        self.mask_ratio=mask_ratio; self.mask_span_length=mask_span_length; self.target_top_k_layers=target_top_k_layers; self.ema_momentum=ema_momentum; self.loss_beta=loss_beta
        self.student_patch=PatchEmbed1D(1,config.embed_dim,patch_size,stride); self.student_encoder=TransformerEncoderBackbone(config); self.mask_token=nn.Parameter(torch.zeros(1,1,config.embed_dim)); self.prediction_head=nn.Linear(config.embed_dim,config.embed_dim)
        self.teacher_patch=copy.deepcopy(self.student_patch); self.teacher_encoder=copy.deepcopy(self.student_encoder)
        for p in list(self.teacher_patch.parameters())+list(self.teacher_encoder.parameters()): p.requires_grad_(False)
        self.teacher_patch.eval(); self.teacher_encoder.eval()
    def train(self,mode=True):
        super().train(mode); self.teacher_patch.eval(); self.teacher_encoder.eval(); return self
    def _teacher_target(self,x):
        with torch.no_grad():
            t,p=self.teacher_patch(x); features=self.teacher_encoder(t,p,return_hidden_states=True); target=torch.stack(features.hidden_states[-self.target_top_k_layers:],dim=0).mean(0); return F.layer_norm(target,(target.shape[-1],),weight=None,bias=None)
    def forward(self,x,*,generator=None,return_prediction=True,return_target=True):
        t,p=self.student_patch(x); b,n,d=t.shape; mask=make_temporal_span_mask(b,n,self.mask_ratio,self.mask_span_length,generator=generator,device=x.device); mask_token=self.mask_token.to(dtype=t.dtype); masked=torch.where(mask.unsqueeze(-1),mask_token.expand(b,n,d),t); pred=self.prediction_head(self.student_encoder(masked,p).tokens); target=self._teacher_target(x); values=(pred-target)[mask]; loss=F.smooth_l1_loss(values,torch.zeros_like(values),beta=self.loss_beta,reduction='mean') if values.numel() else pred.sum()*0.; return Data2VecOutput(loss,mask,pred if return_prediction else None,target if return_target else None,float(mask.float().mean()),target.mean().detach(),target.std(unbiased=False).detach())
    @torch.no_grad()
    def ema_update(self,momentum=None):
        m=self.ema_momentum if momentum is None else momentum
        if not 0<=m<=1: raise ValueError('momentum must be in [0,1]')
        for teacher,student in ((self.teacher_patch,self.student_patch),(self.teacher_encoder,self.student_encoder)):
            for tp,sp in zip(teacher.parameters(),student.parameters()): tp.mul_(m).add_(sp,alpha=1-m)
