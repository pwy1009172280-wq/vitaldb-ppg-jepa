#!/bin/bash
#SBATCH --job-name=mimic-stage1
#SBATCH --account=ttsei19871
#SBATCH --partition=staging
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --output=/scratch-shared/wpu/mimic_stage1_%j.log
#SBATCH --error=/scratch-shared/wpu/mimic_stage1_%j.err

: "${PPG_JEPA_RELEASE_DIR:?submit with PPG_JEPA_RELEASE_DIR pointing to the clean exact-SHA release}"
export PPG_JEPA_CODE_ROOT="$PPG_JEPA_RELEASE_DIR"
export PYTHONPATH="$PPG_JEPA_RELEASE_DIR${PYTHONPATH:+:$PYTHONPATH}"
/scratch-shared/wpu/micromamba-root/envs/ppg-ssl-integration/bin/python -u "$PPG_JEPA_RELEASE_DIR/scripts/mimic_preprocess_stage1.py"
