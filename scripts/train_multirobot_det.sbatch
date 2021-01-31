#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=13
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=multirobot_det

NAME="norobot_det_v7"

python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root /scratch/edward/new_hdf5 --batch_size 64 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --stoch False --epoch_size 400 --checkpoint_interval 30 --g_dim 256 --reconstruction_loss dontcare_mse --last_frame_skip True --scheduled_sampling True --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --video_length 31 --data_threads 4 >"${NAME}.out" 2>&1 &

wait
