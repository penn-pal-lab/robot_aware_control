#!/usr/bin/env bash
#SBATCH --cpus-per-task 4
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=detceminpaintblur

python -m src.mbrl.cem.cem --wandb False --jobname detcemblur1 --multiview True --img_dim 64 --reward_type inpaint-blur --record_trajectory False --topk 5 --action_candidates 30 --opt_iter 5 --horizon 3 --push_dist 0.1 --max_episode_length 3  --large_block True --norobot_pixels_ob True --dynamics_model_ckpt logs/det3svg/ckpt_205000.pt --blur_sigma 1 --stoch False  >detcemblur1.out 2>&1 &

python -m src.mbrl.cem.cem --wandb False --jobname detcemblur3 --multiview True --img_dim 64 --reward_type inpaint-blur --record_trajectory False --topk 5 --action_candidates 30 --opt_iter 5 --horizon 3 --push_dist 0.1 --max_episode_length 3  --large_block True --norobot_pixels_ob True --dynamics_model_ckpt logs/det3svg/ckpt_205000.pt --blur_sigma 3 --stoch False >detcemblur3.out 2>&1  &

#python -m src.mbrl.cem.cem --wandb False --jobname envcemblur3 --multiview True --img_dim 64 --reward_type inpaint-blur --record_trajectory False --topk 5 --action_candidates 30 --opt_iter 5 --horizon 3 --push_dist 0.1 --max_episode_length 3  --large_block True --norobot_pixels_ob True --use_env_dynamics True --blur_sigma 3 --stoch False  >envcemblur3.out 2>&1 &

#python -m src.mbrl.cem.cem --wandb False --jobname envcemblur1 --multiview True --img_dim 64 --reward_type inpaint-blur --record_trajectory False --topk 5 --action_candidates 30 --opt_iter 5 --horizon 3 --push_dist 0.1 --max_episode_length 3  --large_block True --norobot_pixels_ob True --use_env_dynamics True --blur_sigma 1 --stoch False >envcemblur1.out 2>&1  &
wait
