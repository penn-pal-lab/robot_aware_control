#!/usr/bin/env bash
#SBATCH --cpus-per-task 4
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=anonymous-compute
#SBATCH --qos=anonymous-high
#SBATCH --job-name=threshold_sweeps

# Noinpaint img cost only (Baseline)
python -um src.mbrl.episode_runner --wandb True --jobname noinpaint_robotdemo_envcem --multiview True --img_dim 64 --reward_type eef_inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 12000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/redobject_straight_push --demo_type robot_demo  --robot_cost_weight 0 --robot_cost_success 0.02 --red_robot True >noinpaint_robotdemo_envcem.out  2>&1 &

# Threshold Dontcare 6
python -um src.mbrl.episode_runner --wandb True --jobname threshold_dontcare_robotdemo_envcem --multiview True --img_dim 64 --reward_type dontcare  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/redobject_straight_push  --demo_type robot_demo --world_cost_weight 1 --robot_cost_weight 0 --robot_cost_success 0.02 --img_cost_threshold 6 --red_robot True >threshold6_dontcare_robotdemo_envcem.out 2>&1 &

# Threshold Dontcare 10
python -um src.mbrl.episode_runner --wandb True --jobname threshold_dontcare_robotdemo_envcem --multiview True --img_dim 64 --reward_type dontcare  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/redobject_straight_push  --demo_type robot_demo --world_cost_weight 1 --robot_cost_weight 0 --robot_cost_success 0.02 --img_cost_threshold 10 --red_robot True >threshold10_dontcare_robotdemo_envcem.out 2>&1 &

wait
