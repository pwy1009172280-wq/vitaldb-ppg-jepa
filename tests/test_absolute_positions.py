import torch, pytest
from src.models.common.positional_encoding import SinusoidalPositionalEncoding1D
from src.models.common.masking import select_tokens

def test_subset_and_noncontiguous_encoding():
 ids=torch.tensor([[0,2,5,7],[1,3,4,6]]); x=torch.zeros(2,4,8); assert torch.allclose(SinusoidalPositionalEncoding1D(8)(x,ids)[0,1],SinusoidalPositionalEncoding1D(8)(torch.zeros(1,8,8),torch.arange(8))[0,2])
 tokens=torch.randn(2,8,4); mask=torch.tensor([[1,0,1,0,0,1,0,1],[0,1,0,1,1,0,0,1]],dtype=torch.bool); _,p=select_tokens(tokens,torch.arange(8),mask); assert torch.equal(p,torch.tensor([[0,2,5,7],[1,3,4,7]]))

def test_odd_rejected():
 with pytest.raises(ValueError): SinusoidalPositionalEncoding1D(7)
