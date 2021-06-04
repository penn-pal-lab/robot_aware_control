#!/usr/bin/env bash
#SBATCH --cpus-per-task 8
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=ip_modelcem

# ip  model cem
python -um src.mbrl.episode_runner --wandb True --jobname ip_modelcem --multiview True --img_dim 64 --reward_type inpaint --action_candidates 200  --topk 5 --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob True --use_env_dynamics False --num_episodes 100 --most_recent_background False  --world_cost_success 15000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/table_realrobotmask_straight_push --dynamics_model_ckpt logs/mse_prediction2/ckpt_128125.pt --stoch False  --debug_cem False --demo_type object_only_demo --world_cost_weight 1 --robot_cost_weight 0 --robot_cost_success 0.02 --robot_mask_with_obj False --gpu 0 >inpaint_modelcem.out 2>&1 &
wait
