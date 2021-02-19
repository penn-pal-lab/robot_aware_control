#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH -w node-2080ti-5
#SBATCH --job-name=9vp_sawyer

DATA_ROOT="/scratch/edward/Robonet"

# Train on sawyer robot for finetuning to baxter later

# COST="dontcare_mse"
# ACTION="raw"
# NAME="multiviewsawyer_norobot_${ACTION}_imgaug_svg_refactor4"

# python -um src.prediction.multirobot_trainer_refactor --jobname $NAME --wandb True --data_root $DATA_ROOT --lr 0.0003 --batch_size 16 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --model svg --epoch_size 1600 --checkpoint_interval 3 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --training_regime singlerobot  --preprocess_action $ACTION --img_augmentation True >"${NAME}.out" 2>&1 &



# Train norobot model
DATA_ROOT="/scratch/edward/Robonet"
COST="mse"
ACTION="camera_raw"
NAME="sawyerallvp_vanilla_${ACTION}2"

python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --multiview False --image_width 64 --z_dim 64 --g_dim 256  --model svg --niter 1000 --epoch_size 300 --checkpoint_interval 5  --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0003 --training_regime singlerobot --preprocess_action $ACTION --img_augmentation False --world_error_dict baxter_left_world_error.pkl >"${NAME}.out" 2>&1 &

wait