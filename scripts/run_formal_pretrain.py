"""Stage 2 formal pretraining runner (frozen plan §7/§8).

Usage: run_formal_pretrain.py <seed> <windows_dir> <results_root> <run_id>

Builds the 4-block/128-dim JEPA, AdamW + warmup-cosine LR, CUDA AMP + grad clip,
seeded permutation + drop_last, checkpoint every 500 successful updates + step 0,
runs 10,000 successful updates.
"""

import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/gpfs/home2/wpu/projects/vitaldb-ppg-jepa/releases/89915e4fd203111dc5774742893c24333322a652")

from src.config import PipelineConfig
from src.experiments.metadata import seed_everything
from src.experiments.pipeline import build_model_from_config, resolve_device
from src.models.jepa import JEPATrainingAdapter
from src.pretrain.optim import adamw_parameter_groups
from src.training import CheckpointManager, CheckpointSelectionPolicy, Trainer, TrainerConfig
from src.training.schedules import build_warmup_cosine_scheduler

MODEL = {
    "family": "jepa",
    "patch_size": 25, "patch_stride": 25, "embed_dim": 128, "depth": 4,
    "num_heads": 4, "mlp_ratio": 4.0, "dropout": 0.0,
    "decoder_dim": 64, "decoder_depth": 2, "decoder_num_heads": 4, "decoder_mlp_ratio": 4.0,
    "num_target_blocks": 2, "target_block_length": 10, "predictor_dim": 128,
    "predictor_depth": 2, "predictor_num_heads": 4, "predictor_mlp_ratio": 4.0,
    "ema_momentum": 0.996, "loss_beta": 1.0,
}

BATCH_SIZE = 256
MAX_UPDATES = 10000
WARMUP = 500
LR = 1e-4
WEIGHT_DECAY = 0.05
MIN_LR = 1e-6


def main():
    seed = int(sys.argv[1])
    windows_dir = sys.argv[2]
    results_root = sys.argv[3]
    run_id = sys.argv[4] if len(sys.argv) > 4 else f"seed-{seed}"

    seed_everything(seed)
    device = resolve_device("auto")
    print("DEVICE", device, flush=True)

    model = build_model_from_config(MODEL)
    adapter = JEPATrainingAdapter(model)
    adapter.set_mask_generator(torch.Generator().manual_seed(seed + 7919))
    adapter.to(device)

    files = sorted(glob.glob(os.path.join(windows_dir, "*.npy")))
    windows = np.concatenate([np.load(f) for f in files]).astype(np.float32)
    print("WINDOWS", windows.shape, flush=True)
    # seeded permutation (per seed RNG stream)
    rng = np.random.default_rng(seed)
    windows = windows[rng.permutation(len(windows))]
    dataset = TensorDataset(torch.from_numpy(windows))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    optimizer = torch.optim.AdamW(adamw_parameter_groups(adapter, WEIGHT_DECAY), lr=LR)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=WARMUP, total_steps=MAX_UPDATES, min_lr_ratio=MIN_LR / LR)

    resolved = PipelineConfig(model=MODEL, experiment={"mode": "formal", "seed": seed, "device": str(device)})
    from src.experiments.pipeline import _resolved_dict
    from src.experiments.run import create_run

    run, _ = create_run(run_id, _resolved_dict(resolved), "manifest://mimic-formal", "split://pretraining", seed, results_root=results_root)
    manager = CheckpointManager(run.checkpoints_path, CheckpointSelectionPolicy("last"))
    trainer = Trainer(
        adapter, optimizer, scheduler,
        config=TrainerConfig(
            amp=(device.type == "cuda"), grad_clip_norm=1.0, scheduler_step_policy="step",
            max_updates=MAX_UPDATES, checkpoint_interval_updates=500,
        ),
        checkpoint_manager=manager,
        protocol_reference="protocol://pretraining/jepa-formal",
        experiment_manifest_reference=str(run.manifest_path),
        resolved_config=_resolved_dict(resolved),
    )
    trainer.install_signal_handlers()
    # save step 0 (initialization) for audit
    trainer._save_checkpoint("step-0", epoch=0, global_step=0, best_metric_name=None, best_metric_value=None, epoch_complete=False, next_batch_idx=0)
    trainer.fit(loader, epochs=13, max_updates=MAX_UPDATES)
    print("FORMAL_PRETRAIN_DONE seed", seed, "global_step", trainer.global_step, flush=True)


if __name__ == "__main__":
    main()
