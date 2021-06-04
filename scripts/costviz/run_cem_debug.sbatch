#!/usr/bin/env bash
#SBATCH --cpus-per-gpu=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=10G
#SBATCH --time=10:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH -o ./cem_test_30ac.txt
#SBATCH -e ./cem_test_30ac.txt

python -um src.mbrl.cem.cem --wandb True --jobname testwandb7 --img_dim 32 --reward_type dense --action_candidates 10 --topk 3 --horizon 1 --opt_iter 10 --num_episodes 3 --record_trajectory False
