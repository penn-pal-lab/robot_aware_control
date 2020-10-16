#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=collectoptimalactions
#SBATCH -o ./collectoptimalactions.out
#SBATCH -e ./collectoptimalactions.out

python -m src.mbrl.cem.cem --wandb False --jobname collectoptimalactions --multiview True --img_dim 64 --reward_type inpaint --record_trajectory True --action_candidates 30 --opt_iter 10 --horizon 5 --push_dist 0.1 --max_episode_length 5 --large_block True --norobot_pixels_ob True --use_env_dynamics True --record_trajectory_interval 5
