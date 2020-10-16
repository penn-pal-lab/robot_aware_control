#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --mem=150G
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=detsvg256g
#SBATCH -w node-2080ti-5
#SBATCH -o ./detsvg256g.out
#SBATCH -e ./detsvg256g.out

 python -m src.prediction.trainer --jobname det256g  --wandb True --data_root demos/fetch_push_mv --data_threads 0 --batch_size 16 --n_future 4 --n_past 1 --n_eval 5 --multiview True --image_width 64 --z_dim 32 --stoch False --epoch_size 5000 --g_dim 256
