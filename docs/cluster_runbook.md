# Bo cluster runbook

1. Unpack the project package.
2. Create and activate the environment from `envs/formal-ssl-requirements.txt` (or the cluster-approved equivalent).
3. Prepare or locate the processed PPG cache and manifest. Raw CSVs stay outside Git.
4. Fill the cluster placeholders in all three `slurm/*.sbatch` files.
5. Submit the jobs:

```bash
sbatch slurm/train_mae.sbatch
sbatch slurm/train_data2vec.sbatch
sbatch slurm/train_jepa.sbatch
```

`configs/formal/mae.yaml`, `data2vec.yaml`, and `jepa.yaml` are the directly runnable Formal v1 training configurations: 100 epochs, batch size 256, AMP, AdamW, and warmup/cosine scheduling. The `configs/smoke/` files remain optional short diagnostics only. No Python source editing is required.

The Slurm templates override manifest and processed-root paths on the command line. Replace only these cluster-specific placeholders: `<ACCOUNT>`, `<PARTITION>`, `<GPU_REQUEST>`, `<CPUS>`, `<MEMORY>`, `<WALLTIME>`, `<PROJECT_ROOT>`, `ENV_ACTIVATION='<ENV_ACTIVATION>'`, `<PROCESSED_ROOT>`, `<MANIFEST>`, and `<RUN_ROOT>`.

Each run writes `resolved_config.yaml`, `metrics.jsonl`, `train.log`, and `checkpoints/last.pt`. Resume with `python scripts/train_ssl.py --config configs/formal/<method>.yaml --resume <RUN_ROOT>/<method>/checkpoints/last.pt` plus the current data-path overrides. Scientific resume configuration must match the checkpoint; checkpoint format version must be supported.

Inspect the package before transfer with `python scripts/package_for_cluster.py --root . --list-only`; the archive embeds the same `PACKAGE_MANIFEST.txt`. Packaging never uploads or sends files automatically. Perform a real GPU unified-runner smoke before long jobs; individual model CUDA checks have already passed on the RTX 3060.
