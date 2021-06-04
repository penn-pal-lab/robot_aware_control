#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=13
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=multirobot_det
# NAME="norobot_det_16batch_ss"
DATA_ROOT="/scratch/anonymous/Robonet"

# COST="dontcare_mse"
# ACTION="raw"
# NAME="multibot_norobot_${ACTION}_imgaug_fixlr"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --lr 0.0003 --batch_size 16 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --stoch False --epoch_size 1600 --checkpoint_interval 3 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --experiment multirobot  --preprocess_action $ACTION --img_augmentation True >"${NAME}.out" 2>&1 &

# vanilla
COST="mse"
ACTION="raw"
NAME="multibot_vanilla_${ACTION}_imgaug_fixlr"

python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --lr 0.0003 --batch_size 16 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --stoch False --epoch_size 1600 --checkpoint_interval 3 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --experiment multirobot  --preprocess_action $ACTION --img_augmentation True >"${NAME}.out" 2>&1 &

wait