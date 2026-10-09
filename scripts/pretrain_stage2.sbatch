#!/bin/bash
#SBATCH --job-name=jepa-pretrain
#SBATCH --account=ttsei19871
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --output=/scratch-shared/wpu/pretrain_%j.log
#SBATCH --error=/scratch-shared/wpu/pretrain_%j.err

: "${PPG_JEPA_RELEASE_DIR:?submit with PPG_JEPA_RELEASE_DIR pointing to the clean exact-SHA release}"
: "${SEED:?submit with SEED=11 or SEED=29}"
export PPG_JEPA_CODE_ROOT="$PPG_JEPA_RELEASE_DIR"
export PYTHONPATH="$PPG_JEPA_RELEASE_DIR${PYTHONPATH:+:$PYTHONPATH}"
/scratch-shared/wpu/micromamba-root/envs/ppg-ssl-integration/bin/python -u "$PPG_JEPA_RELEASE_DIR/scripts/run_formal_pretrain.py" "$SEED" /scratch-shared/wpu/mimic_pretrain_windows /scratch-shared/wpu/pretrain_results "seed-$SEED"
