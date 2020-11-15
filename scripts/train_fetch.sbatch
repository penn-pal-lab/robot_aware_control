#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=detsvg256g
#SBATCH -w node-2080ti-5

 python -m src.prediction.trainer --jobname cluttersvg  --wandb True --data_root demos/svg_data --data_threads 0 --batch_size 32 --n_future 3 --n_past 1 --n_eval 5 --multiview True --image_width 64 --z_dim 32 --stoch True --epoch_size 5000 --g_dim 256 >cluttersvg.out 2>&1 &
 wait
