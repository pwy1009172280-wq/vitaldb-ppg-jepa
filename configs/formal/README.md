# Formal configuration handoff

The repository ships only short smoke configurations under `configs/smoke/`.
Before long cluster pretraining, an independently reviewed formal YAML must be
created and selected with `--config`; no scientific budget is assumed here.
Freeze batch size, epochs/max_updates, learning rate, weight decay, warmup,
minimum LR ratio, AMP, seeds, checkpoint/log intervals, and method-specific
hyperparameters before submission.
