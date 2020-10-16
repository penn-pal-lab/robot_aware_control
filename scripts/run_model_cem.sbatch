#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=multijob1gpu
#SBATCH -o ./testgpu.out
#SBATCH -e ./testgpu.out

python -m src.mbrl.cem.cem --wandb False --jobname gputest1 --multiview True --img_dim 64 --reward_type inpaint-blur --record_trajectory False --topk 5 --action_candidates 30 --opt_iter 5 --horizon 3 --push_dist 0.1 --max_episode_length 4  --large_block True --norobot_pixels_ob True --dynamics_model_ckpt logs/mvrealseg/ckpt_84600.pt --blur_sigma 1 &

python -m src.mbrl.cem.cem --wandb False --jobname gputest2 --multiview True --img_dim 64 --reward_type inpaint-blur --record_trajectory False --topk 5 --action_candidates 30 --opt_iter 5 --horizon 3 --push_dist 0.1 --max_episode_length 4  --large_block True --norobot_pixels_ob True --dynamics_model_ckpt logs/mvrealseg/ckpt_84600.pt --blur_sigma 1
wait
