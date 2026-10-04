"""Dataset-specific source-native adapters."""

from .ppg_dalia import PPGDaLiAReader
from .ptb_xl import PTBXLReader

__all__ = ["PTBXLReader", "PPGDaLiAReader"]
