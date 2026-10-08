"""Minimal adapter from the standard JEPA module to the generic Trainer.

The adapter owns no scientific logic. It only translates the existing JEPA
forward result and lifecycle method into the TrainingModel contract.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ...training.contracts import LossOutput
from .model import JEPA1D, JEPAOutput, JEPARepresentation


class JEPATrainingAdapter(nn.Module):
    """Expose ``JEPA1D`` through the generic training contract.

    Batches may be a waveform tensor or a mapping containing ``waveform``.
    The wrapped JEPA module remains unchanged; in particular, masking, target
    construction, and Smooth L1 loss stay in ``JEPA1D.forward``.

    A dedicated CPU mask ``generator`` can be attached via
    ``set_mask_generator``; it is forwarded to ``JEPA1D.forward(generator=...)``
    so training mask sampling never consumes the global RNG stream.
    """

    def __init__(self, jepa: JEPA1D, generator: torch.Generator | None = None) -> None:
        super().__init__()
        if not isinstance(jepa, JEPA1D):
            raise TypeError("JEPATrainingAdapter requires a JEPA1D module")
        self.jepa = jepa
        self.generator = generator

    def set_mask_generator(self, generator: torch.Generator | None) -> None:
        self.generator = generator

    @staticmethod
    def _waveform(batch: Any) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            waveform = batch
        elif isinstance(batch, Mapping) and "waveform" in batch:
            waveform = batch["waveform"]
        else:
            raise TypeError("JEPA batches must be a waveform tensor or mapping with 'waveform'")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("batch['waveform'] must be a torch.Tensor")
        return waveform

    def forward(self, batch: Any) -> JEPAOutput:
        waveform = self._waveform(batch)
        if self.generator is not None:
            return self.jepa(waveform, generator=self.generator)
        return self.jepa(waveform)

    def compute_loss(self, output: JEPAOutput, batch: Any) -> LossOutput:
        if not isinstance(output, JEPAOutput):
            raise TypeError("JEPATrainingAdapter.compute_loss expects JEPAOutput")
        return LossOutput(output.loss, {"jepa_loss": output.loss})

    def on_optimizer_step(self) -> None:
        """Apply the existing EMA update after a successful optimizer step."""
        self.jepa.ema_update()

    def encode_full(self, waveform: torch.Tensor) -> JEPARepresentation:
        """Delegate frozen/full-sequence representation extraction."""
        return self.jepa.encode_full(waveform)

    def freeze_encoder(self) -> "JEPATrainingAdapter":
        """Freeze the online representation path for downstream extraction."""
        for module in (self.jepa.online_patch, self.jepa.context_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.eval()
        return self

    def encoder_parameters(self):
        """Return parameters belonging to the online representation encoder."""
        yield from self.jepa.online_patch.parameters()
        yield from self.jepa.context_encoder.parameters()
