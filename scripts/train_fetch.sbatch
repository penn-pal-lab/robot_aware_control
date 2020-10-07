#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --mem=150G
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=mvseg
#SBATCH -o ./mvseg.out
#SBATCH -e ./mvseg.out

 python -m src.prediction.trainer --jobname mvseg --wandb True --data_root demos/fetch_push_mv --data_threads 3 --batch_size 16 --n_future 4 --n_past 1 --n_eval 5 --multiview True --image_width 64

