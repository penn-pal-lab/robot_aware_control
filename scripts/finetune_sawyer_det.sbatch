#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=13
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=multirobot_det

# finetune from norobot model with raw actions
NAME="finetune_norobot_sawyer_raw_det"
COST="mse"
ACTION="raw"
MODEL_PATH="ckpt_108400.pt"

python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root /scratch/edward/new_hdf5 --batch_size 16 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --stoch False --epoch_size 400 --checkpoint_interval 30 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --lr 0.001 --training_regime finetune --preprocess_action $ACTION --dynamics_model_ckpt $MODEL_PATH >"${NAME}.out" 2>&1 &

wait

# finetune from vanilla model with raw actions
# NAME="finetune_vanilla_sawyer_raw_det"
# COST="mse"
# ACTION="raw"
# MODEL_PATH="ckpt_108400.pt"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root /scratch/edward/new_hdf5 --batch_size 16 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --stoch False --epoch_size 400 --checkpoint_interval 30 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --lr 0.001 --training_regime finetune --preprocess_action $ACTION --dynamics_model_ckpt $MODEL_PATH >"${NAME}.out" 2>&1 &

# wait
