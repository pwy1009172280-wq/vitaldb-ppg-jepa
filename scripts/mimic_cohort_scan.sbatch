#!/bin/bash
#SBATCH --job-name=mimic-cohort
#SBATCH --account=ttsei19871
#SBATCH --partition=staging
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --output=/scratch-shared/wpu/mimic_cohort_%j.log
#SBATCH --error=/scratch-shared/wpu/mimic_cohort_%j.err

/scratch-shared/wpu/micromamba-root/envs/ppg-ssl-integration/bin/python -u /scratch-shared/wpu/mimic_cohort.py
