from torch import nn
from .config import BackboneConfig
from .positional_encoding import SinusoidalPositionalEncoding1D
from .types import EncoderFeatures

class TransformerEncoderBackbone(nn.Module):
    def __init__(self,config:BackboneConfig):
        super().__init__(); self.config=config; self.positional_encoding=SinusoidalPositionalEncoding1D(config.embed_dim)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d_model=config.embed_dim,nhead=config.num_heads,dim_feedforward=int(config.embed_dim*config.mlp_ratio),dropout=config.dropout,batch_first=True,activation="gelu") for _ in range(config.depth)])
        self.norm=nn.LayerNorm(config.embed_dim)
    def forward(self,tokens,pos_ids,return_hidden_states=False):
        x=self.positional_encoding(tokens,pos_ids); states=[]
        for layer in self.layers:
            x=layer(x)
            if return_hidden_states: states.append(x)
        return EncoderFeatures(self.norm(x),pos_ids,tuple(states) if return_hidden_states else None)
