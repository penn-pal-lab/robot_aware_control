#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=realisticmask_det
python -m src.prediction.trainer --jobname dontcare_mse_prediction2 --wandb True --data_root demos/table_svg_train --data_threads 0 --batch_size 64 --n_future 3 --n_past 1 --n_eval 3 --multiview True --image_width 64 --z_dim 64 --stoch False --epoch_size 3125 --g_dim 256 --video_type robot_demo --reconstruction_loss dontcare_mse --last_frame_skip True --scheduled_sampling True --robot_pixel_weight 0 >dontcare_mse_prediction.out 2>&1 &

# python -m src.prediction.trainer --jobname mse_prediction2 --wandb True --data_root demos/table_svg_train --data_threads 0 --batch_size 64 --n_future 3 --n_past 1 --n_eval 3 --multiview True --image_width 64 --z_dim 64 --stoch False --epoch_size 3125 --g_dim 256 --video_type robot_demo --reconstruction_loss mse --last_frame_skip True --scheduled_sampling True --robot_pixel_weight 0 >normal_mse_prediction.out 2>&1 &

wait
