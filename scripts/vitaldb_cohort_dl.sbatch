#!/bin/bash
#SBATCH --job-name=vitaldb-cohort-dl
#SBATCH --account=ttsei19871
#SBATCH --partition=staging
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --output=/scratch-shared/wpu/vitaldb_dl_%j.log
#SBATCH --error=/scratch-shared/wpu/vitaldb_dl_%j.err

/scratch-shared/wpu/micromamba-root/envs/ppg-ssl-integration/bin/python -u /scratch-shared/wpu/vitaldb_download_cohort.py
