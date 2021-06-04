#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=inpaintblur_large
#SBATCH -o ./inpaintblur_large.out
#SBATCH -e ./inpaintblur_large.out

python -m src.mbrl.cem.cem --wandb False --jobname inpaintblur_large --img_dim 128 --reward_type inpaint-blur --record_trajectory True --action_candidates 30 --horizon 10 --push_dist 0.2 --max_episode_length 10 --unblur_timestep 1 --unblur_cost_scale 3 --large_block True --opt_iter 2
