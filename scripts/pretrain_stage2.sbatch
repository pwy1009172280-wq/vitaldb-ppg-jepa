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

/scratch-shared/wpu/micromamba-root/envs/ppg-ssl-integration/bin/python -u /scratch-shared/wpu/run_formal_pretrain.py $SEED /scratch-shared/wpu/mimic_pretrain_windows /scratch-shared/wpu/pretrain_results seed-$SEED
