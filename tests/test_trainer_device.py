"""Trainer batch-to-device movement (dict/tensor/list structure handling)."""

import torch
from torch import nn

from src.training.contracts import LossOutput
from src.training.trainer import Trainer, TrainerConfig


class _Passthrough(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return batch

    def compute_loss(self, output, batch):
        return LossOutput(output["waveform"].sum(), {})


def test_move_batch_to_device_preserves_structure():
    model = _Passthrough()
    trainer = Trainer(
        model, torch.optim.AdamW(model.parameters()), config=TrainerConfig(), protocol_reference="test:device"
    )
    batch = {
        "waveform": torch.zeros(2, 1, 16),
        "meta": "keep",
        "nested": [torch.ones(3), 7],
        "tuple": (torch.zeros(2),),
    }
    moved = trainer._move_batch_to_device(batch)
    assert moved["waveform"].device == torch.device("cpu")
    assert moved["meta"] == "keep"
    assert isinstance(moved["nested"], list)
    assert moved["nested"][0].device == torch.device("cpu")
    assert moved["nested"][1] == 7
    assert isinstance(moved["tuple"], tuple)
    assert moved["tuple"][0].device == torch.device("cpu")
