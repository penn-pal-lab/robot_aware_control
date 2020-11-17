#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --mem=30G
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=mrinpaint

 python -m src.prediction.trainer --jobname mr_cluttersvg  --wandb True --data_root demos/svg_mr_inpaint --data_threads 0 --batch_size 32 --n_future 3 --n_past 1 --n_eval 3 --multiview True --image_width 64 --z_dim 32 --stoch True --epoch_size 5000 --g_dim 256 >mr_cluttersvg.out 2>&1 &
 wait
