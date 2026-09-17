#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:<GPU_REQUEST>
#SBATCH --cpus-per-task=<CPUS>
#SBATCH --mem=<MEMORY>
#SBATCH --time=<WALLTIME>
#SBATCH --output=<PROJECT_ROOT>/slurm-data2vec-%j.out
#SBATCH --error=<PROJECT_ROOT>/slurm-data2vec-%j.err
set -euo pipefail
<ENV_ACTIVATION>
python <PROJECT_ROOT>/scripts/train_ssl.py --config <FORMAL_CONFIG> --manifest <MANIFEST> --processed-root <PROCESSED_ROOT> --run-dir <RUN_ROOT>/data2vec
