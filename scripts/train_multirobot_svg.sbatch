#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=conv_svg
DATA_ROOT="/scratch/anonymous/Robonet"

# >>>>>>>>>>>>>>>> vanilla conv SVG
COST="l1"
ACTION="raw"
NAME="sawyerallvp_vanilla3_${ACTION}_convsvg_l1"

python -um src.prediction.multirobot_trainer  --jobname $NAME --wandb True --data_root /scratch/anonymous/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 512 --z_dim 64 --model svg --niter 1000 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5  --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 5 --lr 0.0001 --experiment singlerobot --preprocess_action raw --world_error_dict widowx1_c0_world_error.pkl --train_val_split 0.95 --model_use_robot_state False --model_use_mask False --random_snippet True >"${NAME}.out" 2>&1 &

# >>>>>>>>>>>>>>>>> roboaware conv SVG
# COST="dontcare_l1"
# ACTION="raw"
# NAME="sawyerallvp_norobot3_${ACTION}_svg_l1"

# python -um src.prediction.multirobot_trainer  --jobname $NAME --wandb True --data_root /scratch/anonymous/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 512 --z_dim 64 --model svg --niter 1000 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 5 --lr 0.0001 --experiment singlerobot --preprocess_action raw --world_error_dict widowx1_c0_world_error.pkl --train_val_split 0.95 --model_use_robot_state True --model_use_mask True --model_use_future_mask True --random_snippet True >"${NAME}.out" 2>&1 &

wait