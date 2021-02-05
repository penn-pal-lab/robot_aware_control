#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=13
#SBATCH --gpus=1
#SBATCH --time=40:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=multirobot_det

# Train on sawyer robot for finetuning to baxter later
# NAME="sawyerbot_norobot_det_camerastateinfer"

# python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root /scratch/edward/new_hdf5 --batch_size 16 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --stoch False --epoch_size 1600 --checkpoint_interval 30 --g_dim 256 --reconstruction_loss dontcare_mse --last_frame_skip True --scheduled_sampling True --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --training_regime singlerobot  --preprocess_action camera_state_infer >"${NAME}.out" 2>&1 &
# wait


NAME="sawyerbot_vanilla_det_camerastateinfer"

python -um src.prediction.multirobot_trainer --jobname $NAME --wandb True --data_root /scratch/edward/new_hdf5 --batch_size 16 --n_future 3 --n_past 1 --n_eval 3 --multiview False --image_width 64 --z_dim 64 --stoch False --epoch_size 400 --checkpoint_interval 10 --g_dim 256 --reconstruction_loss mse --last_frame_skip True --scheduled_sampling True --action_dim 5 --action_enc_dim 5 --robot_dim 5 --robot_enc_dim 10 --data_threads 4 --training_regime singlerobot --preprocess_action camera_state_infer >"${NAME}.out" 2>&1 &

wait
