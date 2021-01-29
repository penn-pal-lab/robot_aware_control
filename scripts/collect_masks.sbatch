#!/usr/bin/env bash
#SBATCH --cpus-per-task 12
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=gen_data

python -um src.dataset.collect_mask_data  --gpu 0 >mask_data.out 2>&1 &
wait
