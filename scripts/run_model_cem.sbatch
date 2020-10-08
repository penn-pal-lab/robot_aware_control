#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=64acgoodmodelcem
#SBATCH -o ./64acgoodmodelcem.out
#SBATCH -e ./64acgoodmodelcem.out

python -m src.mbrl.cem.cem --wandb True --jobname 64acgoodmodelcem --multiview True --img_dim 64 --reward_type inpaint --record_trajectory False --action_candidates 64 --opt_iter 10 --horizon 5 --push_dist 0.1 --max_episode_length 10 --large_block True --norobot_pixels_ob True --dynamics_model_ckpt logs/mvrealseg/ckpt_84600.pt
