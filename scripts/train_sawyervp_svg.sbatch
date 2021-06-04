#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=sawyer_vp
#SBATCH -w node-2080ti-5

# Train vanilla model
# >>>>>>>>>>>>>>>> vanilla conv SVG
# COST="l1"
# ACTION="raw"
# NAME="sawyer8vp_vanilla_${ACTION}_svg_l1_128g"

# python -um src.prediction.multirobot_trainer  --jobname $NAME --wandb True --data_root /scratch/anonymous/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 1000 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5  --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment train_sawyer_multiview --preprocess_action $ACTION --train_val_split 0.95 --model_use_robot_state False --model_use_mask False --random_snippet True --world_error_dict sudri2_c1_world_error.pkl >"${NAME}.out" 2>&1 &

# >>>>>>>>>>>>>>>>> roboaware conv SVG
COST="dontcare_l1"
ACTION="raw"
NAME="sawyer8vp_norobot_${ACTION}_svg_l1_128g_nofuturemask"

python -um src.prediction.multirobot_trainer  --jobname $NAME --wandb True --data_root /scratch/anonymous/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 1000 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment train_sawyer_multiview --preprocess_action $ACTION --train_val_split 0.95 --model_use_robot_state True --model_use_mask True --model_use_future_mask False --random_snippet True --world_error_dict sudri2_c1_world_error.pkl >"${NAME}.out" 2>&1 &

wait