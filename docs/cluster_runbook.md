# Cluster runbook

The single-GPU engineering runner is ready. Smoke configs under `configs/smoke/` are intentionally short installation and resume checks; they are not formal scientific experiments. Before long pretraining, create an explicitly reviewed formal YAML and freeze batch size, budget, learning rate, scheduler, AMP, seeds, intervals, and method-specific values. The Slurm templates require that path as `<FORMAL_CONFIG>` and never default to smoke.

Create the formal environment from `envs/formal-ssl-requirements.txt`. Keep raw VitalDB CSVs outside Git under a flat raw root; run `scripts/prepare_vitaldb_ppg.py` and verify the preparation manifest has no failed rows. Smoke locally with `python scripts/train_ssl.py --config configs/smoke/mae.yaml --manifest <MANIFEST> --processed-root <PROCESSED_ROOT> --run-dir <RUN_ROOT>/mae` (and substitute `data2vec` or `jepa`).

Before submission replace `<ACCOUNT>`, `<PARTITION>`, `<GPU_REQUEST>`, `<CPUS>`, `<MEMORY>`, `<WALLTIME>`, `<PROJECT_ROOT>`, `<ENV_ACTIVATION>`, `<FORMAL_CONFIG>`, `<PROCESSED_ROOT>`, `<MANIFEST>`, and `<RUN_ROOT>` in Slurm templates. Output/error files are directly below `<PROJECT_ROOT>`. Submit with `sbatch slurm/train_mae.sbatch`, `train_data2vec.sbatch`, or `train_jepa.sbatch`.

Runs write `resolved_config.yaml`, `metrics.jsonl`, `train.log`, and `checkpoints/last.pt`; resume with `--resume <RUN_ROOT>/<METHOD>/checkpoints/last.pt`. Scientific resume configuration must match strictly and checkpoint `format_version` must be supported; operational path, device, worker, and pin-memory changes may be allowed.

Inspect the exact handoff archive before transfer: `python scripts/package_for_cluster.py --root . --list-only`, then `python scripts/package_for_cluster.py --root . --output /tmp/formal.tar.gz`; verify `PACKAGE_MANIFEST.txt`. Packaging is exact-file allowlist based and never uploads or sends anything automatically. A real-GPU unified-runner smoke for all three methods is recommended before long jobs; individual model GPU smokes have already passed on the local RTX 3060.
