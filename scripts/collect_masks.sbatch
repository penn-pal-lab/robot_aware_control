#!/usr/bin/env bash
#SBATCH --cpus-per-task 12
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=gen_data
#SBATCH -w node-2080ti-5

python -um src.dataset.collect_mask_data  --gpu 0 >mask_data.out 2>&1 &
wait
