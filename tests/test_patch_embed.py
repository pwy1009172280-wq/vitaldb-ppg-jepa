import torch
from src.models.common.patch_embed import PatchEmbed1D

def test_patch_shape_count_and_positions():
 for t,p,s in [(16,4,4),(17,4,4),(17,4,2),(10,3,4)]:
  y,ids=PatchEmbed1D(2,8,p,s)(torch.randn(3,2,t)); n=(t-p)//s+1; assert y.shape==(3,n,8); assert torch.equal(ids,torch.arange(n).repeat(3,1))
