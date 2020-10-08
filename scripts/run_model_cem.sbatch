#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=modelcem
#SBATCH -o ./modelcem.out
#SBATCH -e ./modelcem.out

python -m src.mbrl.cem.cem --wandb True --jobname modelcem --multiview True --img_dim 64 --reward_type inpaint --record_trajectory False --action_candidates 30 --opt_iter 10 --horizon 5 --push_dist 0.1 --max_episode_length 10 --large_block True --norobot_pixels_ob True --dynamics_model_ckpt logs/mvseg/ckpt_57600.pt