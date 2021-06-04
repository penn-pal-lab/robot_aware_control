#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=finetune
#SBATCH -w node-2080ti-5

#>>>>>>>>>>>>>> finetune from vanilla model
COST="l1"
ACTION="raw"
CHECKPOINT="226500"
MODEL_PATH="logs/sawyer8vp_vanilla_${ACTION}_svg_l1_128g/ckpt_${CHECKPOINT}.pt"
NAME="ftsawyervp10step_vanilla_128g_${ACTION}_${CHECKPOINT}_nomask_nostate"

python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 10 --g_dim 256 --z_dim 64 --model svg --niter 1000 --epoch_size 50 --checkpoint_interval 5 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0001 --experiment finetune_sawyer_view --preprocess_action $ACTION --dynamics_model_ckpt $MODEL_PATH --random_snippet True --model_use_mask False --model_use_robot_state False --finetune_num_train 400 --finetune_num_test 400 --world_error_dict sudri2_c1_world_error.pkl >"${NAME}.out" 2>&1 &

# random initialization baseline trained on 400 videos
# COST="mse"
# ACTION="raw"
# NAME="ftsawyervp_hardvideo_vanilla_${ACTION}_randominit_baseline_nomask_nostate"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 10 --multiview False --image_width 64 --z_dim 64 --model svg --epoch_size 300 --checkpoint_interval 1000 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0003 --experiment finetune_sawyer_view --preprocess_action $ACTION --img_augmentation False --model_use_mask False --model_use_robot_state False --finetune_num_train 400 --finetune_num_test 400 --world_error_dict sudri2_c1_world_error.pkl   >"${NAME}.out" 2>&1 &

# random initialization baseline trained on ALL videos
# COST="mse"
# ACTION="raw"
# NAME="ftalldatasawyervp_hardvideo_vanilla_${ACTION}_randominit_baseline_nomask_nostate"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 10 --multiview False --image_width 64 --z_dim 64 --model svg --epoch_size 300 --checkpoint_interval 1000 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling False --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0003 --experiment finetune_sawyer_view --preprocess_action $ACTION --img_augmentation False --model_use_mask False --model_use_robot_state False --finetune_num_train 50000 --finetune_num_test 400 --world_error_dict sudri2_c1_world_error.pkl   >"${NAME}.out" 2>&1 &

# >>>>>>>>>>> finetune from norobot model with raw actions
COST="dontcare_l1"
ACTION="raw"
CHECKPOINT="226500"
MODEL_PATH="logs/sawyer8vp_norobot_${ACTION}_svg_l1_128g_nofuturemask/ckpt_${CHECKPOINT}.pt"
ROBOT_MODEL_PATH="/home/anonymous/log_backup/logs/sawyeralldata_3layer3/ckpt_29100.pt"
NAME="ftsawyervp10step_norobot_128g_${ACTION}_${CHECKPOINT}"

python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root $DATA_ROOT --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 1000 --epoch_size 50 --checkpoint_interval 5 --g_dim 256 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --action_enc_dim 32 --robot_dim 5 --robot_enc_dim 32 --data_threads 4 --lr 0.0001 --experiment finetune_sawyer_view --preprocess_action $ACTION --dynamics_model_ckpt $MODEL_PATH --random_snippet True --model_use_mask True --model_use_robot_state True --model_use_future_mask True  --finetune_num_train 400 --finetune_num_test 400 --world_error_dict sudri2_c1_world_error.pkl --learned_robot_model True --robot_model_ckpt $ROBOT_MODEL_PATH --robot_joint_dim 7 >"${NAME}.out" 2>&1 &
wait
