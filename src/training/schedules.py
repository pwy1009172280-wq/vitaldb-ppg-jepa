"""Learning-rate schedules for the frozen experiment recipe (plan §8)."""

from __future__ import annotations

import math

import torch


def warmup_cosine_lr_lambda(step: int, *, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    """LR factor for a warmup-then-cosine schedule.

    ``step`` is the 0-indexed update counter: the first update uses
    ``1/warmup_steps``, update ``warmup_steps`` reaches 1.0, then cosine decays
    to ``min_lr_ratio`` at ``total_steps``.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_steps <= 0 or total_steps <= 0 or warmup_steps > total_steps:
        raise ValueError("invalid warmup/total steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0,1]")
    fn = lambda step: warmup_cosine_lr_lambda(  # noqa: E731
        step, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
