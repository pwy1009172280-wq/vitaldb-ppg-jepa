# Snellius packaging preparation report

## Scope

Prepared the current repository for a future Snellius transfer. No scientific
design, model code, training semantics, experiment, upload, or Slurm
submission was performed.

## Changes

- Added the already-used `scikit-learn==1.5.2` dependency to
  `envs/formal-ssl-requirements.txt` without changing existing pins.
- Extended `scripts/package_for_cluster.py`'s exact-file allowlist with the
  Gate 2A downstream package, both diagnostic CLIs, its tests, documentation,
  and the prior Gate 2A report.
- The existing formal JEPA configuration remains included at
  `configs/formal/jepa.yaml`.

## Package verification

`package_for_cluster.py --list-only` completed successfully. All requested
Gate 2A files and the existing JEPA/pretraining dependencies were present.
The test archive was created and inspected successfully:

`/tmp/vitaldb-ppg-jepa_gate2a.tar.gz`

The archive contains `PACKAGE_MANIFEST.txt` and the exact allowlisted files.

## Environment dependency verification

Verified with `/home/prowse/miniconda3/envs/ppg-ssl/bin/python`:

- PyTorch 2.6.0+cu124
- NumPy 1.26.4
- SciPy 1.13.1
- pandas 2.2.3
- PyYAML 6.0.2
- tqdm 4.66.5
- pytest 8.3.3
- scikit-learn 1.5.2

All match the requirements file. The system-level `python` command is absent,
but the repository's existing project environment is usable.

## Validation

`git diff --check` passed. No data, checkpoint, credential, or machine-specific
path was added to the archive. Snellius upload and experiments were not run.

## Blockers

No local packaging blocker remains. Actual Snellius execution still requires
cluster-specific account, partition, paths, environment activation, and other
Slurm placeholders to be supplied separately.
