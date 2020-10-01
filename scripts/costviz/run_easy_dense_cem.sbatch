#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=1push_horizon5_dense128_randrobot_nosucc
#SBATCH -o ./1push_horizon5_dense128_randrobot_nosucc.out
#SBATCH -e ./1push_horizon5_dense128_randrobot_nosucc.out

python -m src.mbrl.cem.cem --wandb True --jobname 1push_horizon5_dense128_randrobot_nosucc --img_dim 128 --reward_type dense --record_trajectory True --action_candidates 30 --horizon 5 --push_dist 0.1 --max_episode_length 5
