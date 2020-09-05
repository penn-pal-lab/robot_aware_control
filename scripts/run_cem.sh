#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=10G
#SBATCH --time=10:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH -o ./cem_test_30ac.txt
#SBATCH -e ./cem_test_30ac.txt

python -um src.mbrl.cem.cem --jobname norobot_cem_30ac --img_dim 128 --reward_type weighted --action_candidates 30
