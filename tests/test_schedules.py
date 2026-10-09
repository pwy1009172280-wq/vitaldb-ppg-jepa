"""Tests for the warmup-cosine LR schedule (plan §8)."""

import torch

from src.training.schedules import build_warmup_cosine_scheduler, warmup_cosine_lr_lambda


def test_warmup_linear():
    assert warmup_cosine_lr_lambda(0, warmup_steps=500, total_steps=10000, min_lr_ratio=0.01) == 1 / 500
    assert warmup_cosine_lr_lambda(499, warmup_steps=500, total_steps=10000, min_lr_ratio=0.01) == 1.0


def test_cosine_decay_to_min():
    assert warmup_cosine_lr_lambda(10000, warmup_steps=500, total_steps=10000, min_lr_ratio=0.01) == 0.01
    mid = warmup_cosine_lr_lambda(5250, warmup_steps=500, total_steps=10000, min_lr_ratio=0.01)
    assert 0.01 < mid < 1.0


def test_scheduler_build():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = build_warmup_cosine_scheduler(opt, warmup_steps=500, total_steps=10000, min_lr_ratio=0.01)
    assert sched.get_last_lr()[0] == 1e-4 * (1 / 500)
