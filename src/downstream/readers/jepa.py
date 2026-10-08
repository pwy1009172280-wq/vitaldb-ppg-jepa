"""Reader for the existing JEPA ``encode_full`` representation contract."""

from collections.abc import Mapping

import torch

from ..core.encoder import Representation
from ..core.extraction import FeatureBatch


class JEPARepresentationReader:
    name = "jepa_encode_full"
    version = "1"

    def read(self, model: torch.nn.Module, batch: FeatureBatch) -> Mapping[str, Representation]:
        if not hasattr(model, "encode_full"):
            raise TypeError("JEPA reader requires model.encode_full")
        output = model.encode_full(batch.signal)
        hidden_states = getattr(output, "hidden_states", None)
        final_tokens = getattr(output, "final_tokens", None)
        if hidden_states is None or final_tokens is None:
            raise ValueError("encode_full output must expose hidden_states and final_tokens")
        result = {
            f"layer_{index + 1}": Representation(tensor, "token")
            for index, tensor in enumerate(hidden_states)
        }
        result["final_layer"] = Representation(final_tokens, "token")
        return result


# Short compatibility spelling for callers that prefer a compact reader name.
JEPARRepresentationReader = JEPARepresentationReader
