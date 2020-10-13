#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=inpaintblursig5envcem
#SBATCH -o ./inpaintblursig5envcem.out
#SBATCH -e ./inpaintblursig5envcem.out

python -m src.mbrl.cem.cem --wandb True --jobname inpaintblursig5envcem --multiview True --img_dim 64 --reward_type inpaint-blur --record_trajectory False --action_candidates 30 --opt_iter 10 --horizon 5 --push_dist 0.1 --max_episode_length 10 --large_block True --norobot_pixels_ob True --use_env_dynamics True --blur_sigma 5
