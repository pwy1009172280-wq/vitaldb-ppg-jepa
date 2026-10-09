"""Stage 2 formal pretraining runner (frozen plan §7/§8).

Usage: run_formal_pretrain.py <seed> <stage1_root> <results_root> <run_id>

Builds the 4-block/128-dim JEPA, AdamW + warmup-cosine LR, CUDA AMP + grad clip,
seeded permutation + drop_last, checkpoint every 500 successful updates + step 0,
runs 10,000 successful updates.
"""

import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

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
WINDOW_SHAPE = (225, 2000)


def load_role_windows(stage1_root, role, expected_subjects):
    files = sorted(glob.glob(os.path.join(stage1_root, role, "*.npy")))
    if len(files) != expected_subjects:
        raise RuntimeError(f"expected {expected_subjects} {role} subject arrays; found {len(files)}")
    arrays = []
    for path in files:
        arr = np.load(path, allow_pickle=False)
        if arr.shape != WINDOW_SHAPE or not np.isfinite(arr).all():
            raise RuntimeError(f"invalid {role} subject array: {path}, shape={arr.shape}")
        arrays.append(arr)
    return np.concatenate(arrays).astype(np.float32, copy=False), files


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_health_set(stage1_root, diagnostic_arrays, diagnostic_files):
    manifest_path = os.path.join(stage1_root, "mimic_pretraining_sample_manifest.csv")
    grouped = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["role"] == "diagnostic":
                grouped.setdefault(row["subject_id"], []).append(row)
    by_path = {os.path.basename(path).removesuffix(".npy"): i for i, path in enumerate(diagnostic_files)}
    if len(grouped) != 100 or set(grouped) != set(by_path):
        raise RuntimeError("diagnostic sample manifest does not match the 100 diagnostic arrays")
    selected = []
    selected_rows = []
    for subject_id in sorted(grouped):
        rows = sorted(
            grouped[subject_id],
            key=lambda row: (
                hashlib.sha256(
                    f"first-layerwise-2026-10-09|mimic-health-window|{row['sample_identity']}".encode("utf-8")
                ).hexdigest(),
                row["sample_identity"],
            ),
        )[:10]
        if len(rows) != 10:
            raise RuntimeError(f"expected 10 diagnostic health windows for {subject_id}")
        subject_index = by_path[subject_id]
        arr = np.load(diagnostic_files[subject_index], allow_pickle=False)
        for row in rows:
            index = int(row["window_index"])
            if index < 0 or index >= len(arr):
                raise RuntimeError(f"invalid window index in diagnostic manifest: {row}")
            selected.append(arr[index])
            selected_rows.append({
                "subject_id": subject_id,
                "window_index": index,
                "sample_identity": row["sample_identity"],
                "health_order_sha256": hashlib.sha256(
                    f"first-layerwise-2026-10-09|mimic-health-window|{row['sample_identity']}".encode("utf-8")
                ).hexdigest(),
            })
    if len(selected) != 1000:
        raise RuntimeError(f"expected 1000 fixed diagnostic health windows; found {len(selected)}")
    return np.stack(selected).astype(np.float32, copy=False), selected_rows


class HealthTrainer(Trainer):
    """Run protocol health checks on an RNG-isolated diagnostic stream."""

    def __init__(self, *args, health_windows, health_log_path, diagnostic_seed, **kwargs):
        super().__init__(*args, **kwargs)
        self.health_windows = health_windows
        self.health_log_path = health_log_path
        self.diagnostic_generator = torch.Generator().manual_seed(diagnostic_seed)
        self.previous_degenerate = set()
        self.health_failure = None

    @torch.no_grad()
    def record_health(self):
        device = next(self.model.parameters()).device
        train_generator = self.model.generator
        self.model.set_mask_generator(self.diagnostic_generator)
        self.model.eval()
        losses = []
        representations = {f"layer_{i + 1}": [] for i in range(4)}
        representations["final_layer"] = []
        try:
            for start in range(0, len(self.health_windows), 128):
                batch = torch.from_numpy(self.health_windows[start:start + 128]).to(device).unsqueeze(1)
                output = self.model({"waveform": batch})
                if not torch.isfinite(output.loss):
                    self.health_failure = f"non-finite diagnostic loss at update {self.global_step}"
                    break
                losses.append(float(output.loss.detach().cpu()))
                encoded = self.model.encode_full(batch)
                for index, hidden in enumerate(encoded.hidden_states):
                    representations[f"layer_{index + 1}"].append(hidden.mean(dim=1).float().cpu().numpy())
                representations["final_layer"].append(encoded.final_tokens.mean(dim=1).float().cpu().numpy())
        finally:
            self.model.set_mask_generator(train_generator)
            self.model.train()
        if self.health_failure:
            self.request_stop()
            return
        std_by_layer = {}
        degenerate = set()
        for name, batches in representations.items():
            values = np.concatenate(batches, axis=0)
            if not np.isfinite(values).all():
                self.health_failure = f"non-finite {name} diagnostic representation at update {self.global_step}"
                break
            feature_std = values.std(axis=0, ddof=0)
            std_by_layer[name] = {
                "mean_feature_std": float(feature_std.mean()),
                "minimum_feature_std": float(feature_std.min()),
                "maximum_feature_std": float(feature_std.max()),
            }
            if bool(np.all(feature_std <= 1e-6)):
                degenerate.add(name)
        record = {
            "global_step": self.global_step,
            "diagnostic_jepa_loss": float(np.mean(losses)),
            "pooled_across_window_feature_std": std_by_layer,
            "degenerate_representations": sorted(degenerate),
        }
        with open(self.health_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print("HEALTH", json.dumps(record, sort_keys=True), flush=True)
        repeated = degenerate.intersection(self.previous_degenerate)
        self.previous_degenerate = degenerate
        if repeated:
            self.health_failure = f"REPRESENTATION_DEGENERATE at consecutive checkpoints: {sorted(repeated)}"
            self.request_stop()

    def _step_optimizer(self, accumulation_count):
        did_update = super()._step_optimizer(accumulation_count)
        if did_update and self.global_step > 0 and self.global_step % 1000 == 0:
            self.record_health()
        return did_update


def main():
    seed = int(sys.argv[1])
    stage1_root = sys.argv[2]
    results_root = sys.argv[3]
    run_id = sys.argv[4] if len(sys.argv) > 4 else f"seed-{seed}"

    seed_everything(seed)
    device = resolve_device("auto")
    print("DEVICE", device, flush=True)

    model = build_model_from_config(MODEL)
    adapter = JEPATrainingAdapter(model)
    adapter.set_mask_generator(torch.Generator().manual_seed(seed + 7919))
    adapter.to(device)

    with open(os.path.join(stage1_root, "mimic_stage1_summary.json"), encoding="utf-8") as f:
        stage1_summary = json.load(f)
    if stage1_summary.get("train_subjects") != 900 or stage1_summary.get("diagnostic_subjects") != 100 or stage1_summary.get("short_subjects") != 0:
        raise RuntimeError(f"Stage 1 did not pass the frozen 900/100 gate: {stage1_summary}")
    qc_path = os.path.join(stage1_root, "mimic_stage1_qc.csv")
    sample_manifest_path = os.path.join(stage1_root, "mimic_pretraining_sample_manifest.csv")
    if file_sha256(qc_path) != stage1_summary.get("qc_sha256"):
        raise RuntimeError("Stage 1 QC manifest hash mismatch")
    if file_sha256(sample_manifest_path) != stage1_summary.get("sample_manifest_sha256"):
        raise RuntimeError("Stage 1 sample manifest hash mismatch")
    with open(qc_path, newline="", encoding="utf-8") as f:
        qc_rows = list(csv.DictReader(f))
    qc_by_subject = {row["subject_id"]: row for row in qc_rows}
    if len(qc_rows) != 1000 or len(qc_by_subject) != 1000:
        raise RuntimeError("Stage 1 QC manifest must contain exactly 1000 unique subjects")
    train_windows, train_files = load_role_windows(stage1_root, "train", 900)
    diagnostic_arrays, diagnostic_files = load_role_windows(stage1_root, "diagnostic", 100)
    for role, files in (("train", train_files), ("diagnostic", diagnostic_files)):
        for path in files:
            subject_id = os.path.basename(path).removesuffix(".npy")
            row = qc_by_subject.get(subject_id)
            if row is None or row["role"] != role or row["status"] != "complete" or int(row["selected_windows"]) != 225:
                raise RuntimeError(f"Stage 1 QC does not authorize this {role} array: {path}")
            if file_sha256(path) != row["output_sha256"]:
                raise RuntimeError(f"Stage 1 tensor hash mismatch: {path}")
    health_windows, health_rows = build_health_set(stage1_root, diagnostic_arrays, diagnostic_files)
    windows = train_windows
    print("TRAIN_SUBJECTS", len(train_files), "TRAIN_WINDOWS", windows.shape, flush=True)
    print("DIAGNOSTIC_SUBJECTS", len(diagnostic_files), "HEALTH_WINDOWS", health_windows.shape, flush=True)
    # seeded permutation (per seed RNG stream)
    rng = np.random.default_rng(seed)
    windows = windows[rng.permutation(len(windows))]
    windows = windows[:, None, :]  # [N, 1, 2000] channel dim
    dataset = TensorDataset(torch.from_numpy(windows))
    collate = lambda batch: {"waveform": torch.stack([x[0] for x in batch])}  # noqa: E731
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, collate_fn=collate)

    optimizer = torch.optim.AdamW(adamw_parameter_groups(adapter, WEIGHT_DECAY), lr=LR)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=WARMUP, total_steps=MAX_UPDATES, min_lr_ratio=MIN_LR / LR)

    resolved = PipelineConfig(model=MODEL, experiment={"mode": "formal", "seed": seed, "device": str(device)})
    from src.experiments.pipeline import _resolved_dict
    from src.experiments.run import create_run

    cohort_ref = f"sha256://{stage1_summary['frozen_cohort_sha256']}"
    source_ref = f"sha256://{stage1_summary['sample_manifest_sha256']}"
    run, _ = create_run(
        run_id,
        _resolved_dict(resolved),
        f"mimic-stage1:{source_ref}",
        f"mimic-frozen-cohort:{cohort_ref}",
        seed,
        results_root=results_root,
        pretraining_subject_ids=tuple(
            sorted(
                [os.path.basename(path).removesuffix(".npy") for path in train_files]
                + [os.path.basename(path).removesuffix(".npy") for path in diagnostic_files]
            )
        ),
    )
    health_rows_path = run.logs_path / "diagnostic_health_windows.csv"
    with open(health_rows_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "window_index", "sample_identity", "health_order_sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(health_rows)
    manager = CheckpointManager(run.checkpoints_path, CheckpointSelectionPolicy("last"))
    trainer = HealthTrainer(
        adapter, optimizer, scheduler,
        config=TrainerConfig(
            amp=(device.type == "cuda"), grad_clip_norm=1.0, scheduler_step_policy="step",
            max_updates=MAX_UPDATES, checkpoint_interval_updates=500,
        ),
        checkpoint_manager=manager,
        protocol_reference="protocol://pretraining/jepa-formal",
        experiment_manifest_reference=str(run.manifest_path),
        resolved_config=_resolved_dict(resolved),
        health_windows=health_windows,
        health_log_path=str(run.logs_path / "health.jsonl"),
        diagnostic_seed=seed + 15485863,
    )
    trainer.install_signal_handlers()
    # save step 0 (initialization) for audit
    trainer.record_health()
    if trainer.health_failure:
        raise SystemExit(trainer.health_failure)
    trainer._save_checkpoint("step-0", epoch=0, global_step=0, best_metric_name=None, best_metric_value=None, epoch_complete=False, next_batch_idx=0)
    trainer.fit(loader, epochs=13, max_updates=MAX_UPDATES)
    if trainer.health_failure:
        raise SystemExit(trainer.health_failure)
    if trainer.global_step != MAX_UPDATES:
        raise SystemExit(f"formal pretraining ended at {trainer.global_step}, expected {MAX_UPDATES} successful updates")
    print("FORMAL_PRETRAIN_DONE seed", seed, "global_step", trainer.global_step, flush=True)


if __name__ == "__main__":
    main()
