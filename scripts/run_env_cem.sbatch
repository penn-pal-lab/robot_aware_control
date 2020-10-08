#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH -w nvidia-2080ti-5
#SBATCH --job-name=30acenvcem
#SBATCH -o ./30acenvcem.out
#SBATCH -e ./30acenvcem.out

python -m src.mbrl.cem.cem --wandb True --jobname 30acenvcem --multiview True --img_dim 64 --reward_type inpaint --record_trajectory False --action_candidates 30 --opt_iter 10 --horizon 5 --push_dist 0.1 --max_episode_length 10 --large_block True --norobot_pixels_ob True --use_env_dynamics True
