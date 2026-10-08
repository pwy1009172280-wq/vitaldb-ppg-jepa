"""Small benchmark-only 1D SSL baselines; not final research implementations."""
from __future__ import annotations
import copy
import torch
from torch import nn
import torch.nn.functional as F
from .common import BackboneConfig, PatchEmbed1D, TransformerEncoderBackbone
from .common.masking import make_temporal_mask, select_tokens
from .jepa.masking import make_jepa_masks

PATCH = 50

def _encoder(config):
    return nn.ModuleDict({"patch": PatchEmbed1D(1, config.embed_dim, PATCH, PATCH),
                          "backbone": TransformerEncoderBackbone(config)})

def _ema(target, source, momentum):
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.mul_(momentum).add_(s, alpha=1 - momentum)

class MAE1D(nn.Module):
    """75% random patch masking; loss is MSE over raw masked patches only."""
    def __init__(self, config: BackboneConfig, mask_ratio=.75):
        super().__init__(); self.mask_ratio=mask_ratio; self.encoder=_encoder(config)
        dc=BackboneConfig(64, 2, 4, 4, 0., "sinusoidal")
        self.to_decoder=nn.Linear(config.embed_dim,64); self.mask_token=nn.Parameter(torch.zeros(1,1,64))
        self.decoder=TransformerEncoderBackbone(dc); self.head=nn.Linear(64,PATCH)
    def forward(self,x):
        tok,pos=self.encoder["patch"](x); b,n,d=tok.shape; mask=make_temporal_mask(b,n,self.mask_ratio,device=x.device)
        visible, vpos=select_tokens(tok,pos,~mask); encoded=self.encoder["backbone"](visible,vpos).tokens
        decoded=self.to_decoder(encoded); full=self.mask_token.to(decoded).expand(b,n,-1).clone(); full[~mask]=decoded.reshape(-1,64)
        pred=self.head(self.decoder(full,pos).tokens)
        target=x.unfold(-1,PATCH,PATCH).squeeze(1)
        return F.mse_loss(pred[mask],target[mask]), {"prediction":pred,"mask":mask}

class Data2VecStyle1D(nn.Module):
    """Student has learned masked token inputs. Teacher is full-input EMA; targets
    are the mean of its final two pre-final-norm block outputs, LayerNormed over D."""
    def __init__(self, config: BackboneConfig, mask_ratio=.65, ema=.996):
        super().__init__(); self.mask_ratio=mask_ratio; self.ema=ema; self.student=_encoder(config)
        self.teacher=copy.deepcopy(self.student); self.mask_token=nn.Parameter(torch.zeros(1,1,config.embed_dim))
        for p in self.teacher.parameters(): p.requires_grad_(False)
    def forward(self,x):
        tok,pos=self.student["patch"](x); b,n,d=tok.shape; mask=make_temporal_mask(b,n,self.mask_ratio,device=x.device)
        masked=torch.where(mask.unsqueeze(-1),self.mask_token.to(tok).expand(b,n,-1),tok)
        out=self.student["backbone"](masked,pos).tokens
        with torch.no_grad():
            tt,tp=self.teacher["patch"](x); hs=self.teacher["backbone"](tt,tp,True).hidden_states
            target=F.layer_norm(torch.stack(hs[-2:]).mean(0),(d,))
        return F.mse_loss(out[mask],target[mask]), {"prediction":out,"mask":mask,"target":target}
    def update_ema(self): _ema(self.teacher,self.student,self.ema)

class JEPA1D(nn.Module):
    """Vanilla latent JEPA: disjoint 50% context / 25% target positions. Target
    encoder sees full input; predictor fills target queries using positional IDs."""
    def __init__(self, config: BackboneConfig, context_ratio=.50, target_ratio=.25, ema=.996):
        super().__init__(); self.context_ratio=context_ratio; self.target_ratio=target_ratio; self.ema=ema
        self.context=_encoder(config); self.target=copy.deepcopy(self.context)
        for p in self.target.parameters(): p.requires_grad_(False)
        self.target_query=nn.Parameter(torch.zeros(1,1,config.embed_dim)); self.predictor=TransformerEncoderBackbone(config)
    def forward(self,x):
        tok,pos=self.context["patch"](x); b,n,d=tok.shape; masks=make_jepa_masks(b,n,self.context_ratio,self.target_ratio,device=x.device)
        ctx,cpos=select_tokens(tok,pos,masks.context_mask); ctx=self.context["backbone"](ctx,cpos).tokens
        full=self.target_query.to(ctx).expand(b,n,-1).clone(); full[masks.context_mask]=ctx.reshape(-1,d)
        pred=self.predictor(full,pos).tokens; pred,_=select_tokens(pred,pos,masks.target_mask)
        with torch.no_grad():
            tt,tp=self.target["patch"](x); target=self.target["backbone"](tt,tp).tokens; target,_=select_tokens(target,tp,masks.target_mask)
        return F.mse_loss(pred,target), {"prediction":pred,"target":target,"context_mask":masks.context_mask,"target_mask":masks.target_mask}
    def update_ema(self): _ema(self.target,self.context,self.ema)

def build_method(name, config=None):
    config=config or BackboneConfig()
    return {"mae":MAE1D,"data2vec":Data2VecStyle1D,"jepa":JEPA1D}[name](config)
