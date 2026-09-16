import torch
from torch import nn

class SinusoidalPositionalEncoding1D(nn.Module):
    def __init__(self,embed_dim:int):
        super().__init__()
        if embed_dim<=0: raise ValueError("embed_dim must be > 0")
        if embed_dim%2: raise ValueError("embed_dim must be even for sinusoidal encoding")
        self.embed_dim=embed_dim
    def forward(self,tokens:torch.Tensor,pos_ids:torch.Tensor):
        if tokens.ndim!=3: raise ValueError("tokens must have shape [B,N,D]")
        if tokens.shape[-1]!=self.embed_dim: raise ValueError("token embedding dimension does not match positional encoding")
        if pos_ids.ndim==1: pos_ids=pos_ids.unsqueeze(0)
        if pos_ids.ndim!=2 or pos_ids.shape[1]!=tokens.shape[1] or pos_ids.shape[0] not in (1,tokens.shape[0]): raise ValueError("pos_ids must be [N] or [B,N]")
        pos=pos_ids.to(device=tokens.device,dtype=tokens.dtype).unsqueeze(-1)
        half=self.embed_dim//2; scale=-torch.log(torch.tensor(10000.,device=tokens.device,dtype=tokens.dtype))/half
        div=torch.exp(torch.arange(half,device=tokens.device,dtype=tokens.dtype)*scale)
        angles=pos*div
        return tokens+torch.cat((angles.sin(),angles.cos()),dim=-1)
