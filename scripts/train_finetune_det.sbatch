#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=13
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH -w node-2080ti-5
#SBATCH --job-name=multirobot_det

DATA_ROOT="/scratch/edward/Robonet"

# Train on sawyer robot for finetuning to baxter later

# COST="dontcare_mse"
# ACTION="raw"
# NAME="multiviewsawyer_norobot_${ACTION}_imgaug_svg_refactor4"

# python -um src.prediction.multirobot_trainer_refactor --jobname $NAME --wandb True --data_root $DATA_ROOT --lr 0.0003 --batch_size 16 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --model svg --epoch_size 1600 --checkpoint_interval 3 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --training_regime singlerobot  --preprocess_action $ACTION --img_augmentation True >"${NAME}.out" 2>&1 &



COST="mse"
ACTION="raw"
NAME="multiviewsawyer_vanilla_${ACTION}_imgaug_fixlr_refactor4"

python -um src.prediction.multirobot_trainer_refactor --jobname $NAME --wandb True --data_root $DATA_ROOT --lr 0.0003 --batch_size 16 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --model det --epoch_size 1600 --checkpoint_interval 3000 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --training_regime singlerobot  --preprocess_action $ACTION --img_augmentation True >"${NAME}.out" 2>&1 &

wait
