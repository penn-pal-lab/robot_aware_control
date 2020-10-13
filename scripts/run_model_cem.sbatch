#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH -w node-2080ti-5
#SBATCH --job-name=inpaintblursig5push2
#SBATCH -o ./inpaintblursig5push2.out
#SBATCH -e ./inpaintblursig5push2.out

python -m src.mbrl.cem.cem --wandb True --jobname inpaintblursig5push2 --multiview True --img_dim 64 --reward_type inpaint-blur --record_trajectory False --action_candidates 30 --opt_iter 10 --horizon 10 --push_dist 0.2 --max_episode_length 10  --large_block True --norobot_pixels_ob True --dynamics_model_ckpt logs/mvrealseg/ckpt_84600.pt --blur_sigma 5
