#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=skip_robot_ssl2mask_det
python -m src.prediction.trainer --jobname realvp_randomdata_noinpaint_det --wandb True --data_root demos/realvp_random_mask_svg_train --data_threads 0 --batch_size 64 --n_future 3 --n_past 1 --n_eval 3 --multiview True --image_width 64 --z_dim 64 --stoch False --epoch_size 3125 --g_dim 256 --video_type robot_demo --reconstruction_loss mse --last_frame_skip True --scheduled_sampling True >realvp_random_noinpaint_det.out 2>&1 &
 wait
