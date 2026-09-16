import torch
from torch import nn

class PatchEmbed1D(nn.Module):
    """Conv1d projection: [B,C,T] -> [B,N,D]. No padding; trailing remainder is dropped."""
    def __init__(self, in_channels:int, embed_dim:int, patch_size:int, patch_stride:int|None=None):
        super().__init__()
        if in_channels<=0: raise ValueError("in_channels must be > 0")
        if patch_size<=0: raise ValueError("patch_size must be > 0")
        self.patch_size=patch_size; self.patch_stride=patch_size if patch_stride is None else patch_stride
        if self.patch_stride<=0: raise ValueError("patch_stride must be > 0")
        self.proj=nn.Conv1d(in_channels,embed_dim,patch_size,stride=self.patch_stride,padding=0)
    def forward(self,x:torch.Tensor):
        if x.ndim!=3: raise ValueError(f"expected [B,C,T], got {tuple(x.shape)}")
        if x.shape[-1]<self.patch_size: raise ValueError("input length is shorter than patch_size")
        tokens=self.proj(x).transpose(1,2)
        n=tokens.shape[1]
        pos_ids=torch.arange(n,device=x.device,dtype=torch.long).unsqueeze(0).expand(x.shape[0],-1).contiguous()
        return tokens,pos_ids
