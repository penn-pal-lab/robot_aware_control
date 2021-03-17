#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=conv_svg
DATA_ROOT="/scratch/edward/Robonet"

# vanilla conv SVG
COST="l1"
ACTION="raw"
NAME="sawyerallvp_vanilla_${ACTION}_convdet_l1"

python -um --jobname $NAME --wandb True --data_root /scratch/edward/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --multiview False --image_width 64 --g_dim 512 ---z_dim 64 --model svg --niter 1000 --epoch_size 1500 --checkpoint_interval 5 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 5 --lr 0.0001 --training_regime singlerobot --preprocess_action raw --world_error_dict widowx1_c0_world_error.pkl --train_val_split 0.95 --model_use_robot_state False --model_use_mask False --random_snippet True >"${NAME}.out" 2>&1 &

wait