#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=10G
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH -o ./cem_test.txt
#SBATCH -e ./cem_test.txt
python -m src.mbrl.cem.cem --img_dim 32 --reward_type weighted