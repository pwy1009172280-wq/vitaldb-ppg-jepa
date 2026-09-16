import torch, pytest
from src.models.common import BackboneConfig,TransformerEncoderBackbone

def test_backbone_hidden_states_and_positions():
 c=BackboneConfig(embed_dim=16,depth=3,num_heads=4,mlp_ratio=2); o=TransformerEncoderBackbone(c)(torch.randn(2,4,16),torch.tensor([[0,2,5,7],[1,3,4,9]]),True); assert o.tokens.shape==(2,4,16) and len(o.hidden_states)==3 and all(h.shape==(2,4,16) for h in o.hidden_states) and torch.isfinite(o.tokens).all()

def test_backbone_cuda():
 if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
 m=TransformerEncoderBackbone(BackboneConfig(embed_dim=16,depth=2,num_heads=4)).cuda(); o=m(torch.randn(2,4,16,device='cuda'),torch.tensor([[0,2,5,7],[1,3,4,9]],device='cuda'),True); assert o.tokens.is_cuda and torch.isfinite(o.tokens).all() and all(h.is_cuda for h in o.hidden_states)

def test_config_validation():
 with pytest.raises(ValueError): BackboneConfig(embed_dim=7,num_heads=1)
 with pytest.raises(ValueError): BackboneConfig(embed_dim=15,num_heads=2)
 with pytest.raises(ValueError): BackboneConfig(depth=0)
 with pytest.raises(ValueError): BackboneConfig(dropout=1)
