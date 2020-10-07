#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --mem=150G
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=acprior
#SBATCH -o ./acprior.out
#SBATCH -e ./acprior.out

 python -m src.prediction.trainer --jobname acprior2 --wandb True --data_root demos/fetch_push --data_threads 3 --batch_size 16 --n_future 4 --n_past 1 --n_eval 5

