#!/bin/bash
#SBATCH --job-name=vitaldb-stage4
#SBATCH --account=ttsei19871
#SBATCH --partition=staging
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --output=/scratch-shared/wpu/vitaldb_stage4_%j.log
#SBATCH --error=/scratch-shared/wpu/vitaldb_stage4_%j.err

/scratch-shared/wpu/micromamba-root/envs/ppg-ssl-integration/bin/python -u /scratch-shared/wpu/vitaldb_preprocess_stage4.py
