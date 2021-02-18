#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=13
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=sawyer_vp
#SBATCH -w node-2080ti-5

# Train vanilla model
# DATA_ROOT="/scratch/edward/Robonet"
# COST="mse"
# ACTION="raw"
# NAME="sawyer8vp_vanilla_${ACTION}"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 5 --multiview False --image_width 64 --z_dim 64 --model svg --niter 1000 --epoch_size 300 --checkpoint_interval 5 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --lr 0.0003 --training_regime train_sawyer_multiview --preprocess_action $ACTION --img_augmentation False >"${NAME}.out" 2>&1 &

# Train norobot model
DATA_ROOT="/scratch/edward/Robonet"
COST="mse"
ACTION="camera_raw"
NAME="sawyer8vp_vanilla_${ACTION}7"

python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 5 --multiview False --image_width 64 --z_dim 64 --g_dim 256  --model svg --niter 1000 --epoch_size 300 --checkpoint_interval 5  --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0003 --training_regime train_sawyer_multiview --preprocess_action $ACTION --img_augmentation False >"${NAME}.out" 2>&1 &

wait
