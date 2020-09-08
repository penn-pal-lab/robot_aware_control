#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=weighted128_randrobot


python -m src.mbrl.cem.cem --wandb True --jobname weighted128_randrobot --img_dim 128 --reward_type weighted --record_trajectory True
