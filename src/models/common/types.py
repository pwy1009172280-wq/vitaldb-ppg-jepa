from dataclasses import dataclass
import torch

@dataclass
class EncoderFeatures:
    """Final tokens are post-LayerNorm; hidden_states are per-block pre-final-norm outputs."""
    tokens: torch.Tensor
    pos_ids: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
