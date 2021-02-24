#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=finetune
#SBATCH -w node-2080ti-5


DATA_ROOT="/scratch/edward/Robonet"
# finetune from vanilla model
COST="mse"
ACTION="raw"
CHECKPOINT="42300"
MODEL_PATH="logs/sawyer8vp_vanilla_raw_nomask_nostate/ckpt_${CHECKPOINT}.pt"
NAME="ftsawyervp_hardvideo_vanilla_${ACTION}_${CHECKPOINT}_nomask_nostate"

python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 10 --multiview False --image_width 64 --z_dim 64 --model svg --epoch_size 300 --checkpoint_interval 1000 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0003 --training_regime finetune_sawyer_view --preprocess_action $ACTION --dynamics_model_ckpt $MODEL_PATH --img_augmentation False --model_use_mask False --model_use_robot_state False --finetune_num_train 400 --finetune_num_test 400 --world_error_dict sudri2_c1_world_error.pkl  >"${NAME}.out" 2>&1 &

# random initialization baseline trained on 400 videos
# COST="mse"
# ACTION="raw"
# NAME="ftsawyervp_hardvideo_vanilla_${ACTION}_randominit_baseline_nomask_nostate"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 10 --multiview False --image_width 64 --z_dim 64 --model svg --epoch_size 300 --checkpoint_interval 1000 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0003 --training_regime finetune_sawyer_view --preprocess_action $ACTION --img_augmentation False --model_use_mask False --model_use_robot_state False --finetune_num_train 400 --finetune_num_test 400 --world_error_dict sudri2_c1_world_error.pkl   >"${NAME}.out" 2>&1 &

# random initialization baseline trained on ALL videos
# COST="mse"
# ACTION="raw"
# NAME="ftalldatasawyervp_hardvideo_vanilla_${ACTION}_randominit_baseline_nomask_nostate"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 10 --multiview False --image_width 64 --z_dim 64 --model svg --epoch_size 300 --checkpoint_interval 1000 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0003 --training_regime finetune_sawyer_view --preprocess_action $ACTION --img_augmentation False --model_use_mask False --model_use_robot_state False --finetune_num_train 50000 --finetune_num_test 400 --world_error_dict sudri2_c1_world_error.pkl   >"${NAME}.out" 2>&1 &

# finetune from norobot model with raw actions
# COST="dontcare_mse"
# ACTION="raw"
# CHECKPOINT="42300"
# MODEL_PATH="/home/edward/log_backup/logs/sawyer8vp_norobot_raw_fixrng/ckpt_${CHECKPOINT}.pt"
# NAME="ftsawyervp_hardvideo_norobot_${ACTION}_${CHECKPOINT}"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 10 --multiview False --image_width 64 --z_dim 64 --model svg --epoch_size 300 --checkpoint_interval 1000 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0003 --training_regime finetune_sawyer_view --preprocess_action $ACTION --dynamics_model_ckpt $MODEL_PATH --img_augmentation False --finetune_num_train 400 --finetune_num_test 400  --world_error_dict sudri2_c1_world_error.pkl >"${NAME}.out" 2>&1 &

wait
