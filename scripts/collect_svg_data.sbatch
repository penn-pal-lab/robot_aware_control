#!/usr/bin/env bash
#SBATCH --cpus-per-task 12
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=democem

python -um src.dataset.collect_clutter_data --gpu 0 >svg_data.out 2>&1 &
wait
