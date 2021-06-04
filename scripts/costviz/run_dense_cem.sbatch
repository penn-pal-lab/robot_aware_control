#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=dense128_randrobot
#SBATCH -o ./dense128_randrobot.out
#SBATCH -e ./dense128_randrobot.out

python -m src.mbrl.cem.cem --wandb True --jobname dense128_randrobot_30ac --img_dim 128 --reward_type dense --record_trajectory True --action_candidates 30