#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=2push_horizon10_weighted128_randrobot_nosucc
#SBATCH -o ./2push_horizon10_weighted128_randrobot_nosucc.out
#SBATCH -e ./2push_horizon10_weighted128_randrobot_nosucc.out

python -m src.mbrl.cem.cem --wandb True --jobname 2push_horizon10_weighted128_randrobot_nosucc --img_dim 128 --reward_type weighted --record_trajectory True --action_candidates 30 --horizon 10 --push_dist 0.2 --max_episode_length 10
