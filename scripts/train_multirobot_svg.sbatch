#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=multirobot_det

python -m src.prediction.multirobot_trainer --jobname normal_mse_prediction --wandb False --data_root /home/ed/Robonet/hdf5 --batch_size 12 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --stoch False --epoch_size 3125 --g_dim 256 --reconstruction_loss mse --last_frame_skip True --scheduled_sampling True --action_dim 5 --action_enc_dim 5  --robot_dim 5  --robot_enc_dim 5 --video_length 31

 >normal_mse_prediction.out 2>&1 &

wait
