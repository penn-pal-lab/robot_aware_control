#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --job-name=cost_pick_vfs
#SBATCH -w node-2080ti-5


python -m src.prediction.trainer --jobname cost_pick_vfs --wandb True --data_root /scratch/edward/Robonet --batch_size 16 --n_future 10 --n_past 1 --n_eval 11 --g_dim 256 --z_dim 64 --model svg --niter 300 --epoch_size 300 --eval_interval 15 --checkpoint_interval 40 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment train_locobot_pick --preprocess_action raw --video_length 15 --model_use_robot_state True --model_use_mask False --model_use_future_mask False --model_use_future_robot_state False --random_snippet True --wandb_group pick5_train --lstm_group_norm True --use_skip False >cost_pick_vfs.out 2>&1 &

wait
