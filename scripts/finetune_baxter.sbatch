#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=finetune
#SBATCH -w node-2080ti-5


DATA_ROOT="/scratch/anonymous/Robonet"
# finetune from vanilla model
# COST="mse"
# ACTION="raw"
# CHECKPOINT="35200"
# MODEL_PATH="logs/multiviewsawyer_vanilla_raw_imgaug_svg/ckpt_${CHECKPOINT}.pt"
# NAME="ftbaxter_vanilla_${ACTION}_${CHECKPOINT}_imgaug_svg3"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --stoch True --epoch_size 400 --checkpoint_interval 30 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --lr 0.0003 --experiment finetune --preprocess_action $ACTION --dynamics_model_ckpt $MODEL_PATH --img_augmentation True --test_batch_size 16 >"${NAME}.out" 2>&1 &

# random initialization baseline
COST="mse"
ACTION="raw"
NAME="ftmotion500baxter_vanilla_${ACTION}_randominit_baseline_nomask_nostate_fixed"

python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --multiview False --image_width 64 --z_dim 64 --model svg --epoch_size 300 --checkpoint_interval 1000 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0003 --experiment finetune --preprocess_action $ACTION --img_augmentation False --model_use_mask False --model_use_robot_state False --world_error_dict baxter_left_world_error.pkl  >"${NAME}.out" 2>&1 &

# finetune from norobot model with raw actions
# COST="dontcare_mse"
# ACTION="raw"
# CHECKPOINT="42300"
# MODEL_PATH="logs/sawyerallvp_norobot_raw2/ckpt_${CHECKPOINT}.pt"
# NAME="ftmotion500baxter_norobot_${ACTION}_${CHECKPOINT}"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --multiview False --image_width 64 --z_dim 64 --model svg --epoch_size 300 --checkpoint_interval 1000 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0003 --experiment finetune --preprocess_action $ACTION --img_augmentation False --dynamics_model_ckpt $MODEL_PATH --world_error_dict baxter_left_world_error.pkl  >"${NAME}.out" 2>&1 &

wait
