#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=10G
#SBATCH --time=10:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH -o ./pixel_cem_test.txt
#SBATCH -e ./pixel_cem_test.txt

python -um src.mbrl.cem.cem --prefix pixel_cem --img_dim 128 --reward_type dense --action_candidates 60
