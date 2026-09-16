from dataclasses import dataclass

@dataclass(frozen=True)
class BackboneConfig:
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    positional_encoding: str = "sinusoidal"

    def __post_init__(self):
        if self.embed_dim <= 0: raise ValueError("embed_dim must be > 0")
        if self.depth <= 0: raise ValueError("depth must be > 0")
        if self.num_heads <= 0: raise ValueError("num_heads must be > 0")
        if self.embed_dim % 2: raise ValueError("embed_dim must be even for sinusoidal encoding")
        if self.embed_dim % self.num_heads: raise ValueError("embed_dim must be divisible by num_heads")
        if self.mlp_ratio <= 0: raise ValueError("mlp_ratio must be > 0")
        if not 0 <= self.dropout < 1: raise ValueError("dropout must be in [0, 1)")
        if self.positional_encoding != "sinusoidal": raise ValueError("unsupported positional_encoding")
