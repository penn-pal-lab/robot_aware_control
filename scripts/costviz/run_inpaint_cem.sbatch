#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=inpaint_weighted128_randrobot
#SBATCH -o ./inpaint_weighted128_randrobot.out
#SBATCH -e ./inpaint_weighted128_randrobot.out

python -m src.mbrl.cem.cem --wandb True --jobname inpaint_h5_weighted128_randrobot_nosucc --img_dim 128 --reward_type weighted --record_trajectory False --action_candidates 30 --horizon 5 --push_dist 0.2 --max_episode_length 10 --inpaint True
