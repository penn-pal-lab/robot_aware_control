#!/usr/bin/env bash
#SBATCH --cpus-per-task 4
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=redrobotobject_costs

# Threshold Dontcare only
python -um src.mbrl.episode_runner --wandb True --jobname threshold_dontcare_robotdemo_envcem --multiview True --img_dim 64 --reward_type dontcare  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/redobject_straight_push  --demo_type robot_demo --world_cost_weight 1 --robot_cost_weight 0 --robot_cost_success 0.02 --img_cost_threshold 3 --red_robot True >threshold_dontcare_robotdemo_envcem.out 2>&1 &

# Dontcare only
python -um src.mbrl.episode_runner --wandb True --jobname dontcare_robotdemo_envcem --multiview True --img_dim 64 --reward_type dontcare  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/redobject_straight_push  --demo_type robot_demo --world_cost_weight 1 --robot_cost_weight 0 --robot_cost_success 0.02 --red_robot True >dontcare_robotdemo_envcem.out 2>&1 &

# Robot cost + Inpainted image cost on inpainted demonstration of robot pushing (ours)
python -um src.mbrl.episode_runner --wandb True --jobname eefinpaint_robotdemo_envcem --multiview True --img_dim 64 --reward_type eef_inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob True  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 12000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/redobject_straight_push --demo_type robot_demo  --robot_cost_weight 50000 --robot_cost_success 0.02 --red_robot True >eefinpaint_robotdemo_envcem.out  2>&1 &

# Robot Cost + Threshold Dontcare image cost (ours)
python -um src.mbrl.episode_runner --wandb True --jobname eefthresholddontcare_robotdemo_envcem --multiview True --img_dim 64 --reward_type dontcare  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/redobject_straight_push  --demo_type robot_demo --world_cost_weight 1 --robot_cost_weight 1 --robot_cost_success 0.02 --img_cost_threshold 3 --red_robot True >eefthresholddontcare_robotdemo_envcem.out 2>&1 &

# Robot Cost + Dontcare image cost (ours)
python -um src.mbrl.episode_runner --wandb True --jobname eefdontcare_robotdemo_envcem --multiview True --img_dim 64 --reward_type dontcare  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/redobject_straight_push  --demo_type robot_demo --world_cost_weight 1 --robot_cost_weight 1 --robot_cost_success 0.02 --red_robot True >eefdontcare_robotdemo_envcem.out 2>&1 &

# Robot cost + No-inpaint image cost on demonstration of robot pushing (baseline)
python -um src.mbrl.episode_runner --wandb True --jobname eefnoinpaint_robotdemo_envcem --multiview True --img_dim 64 --reward_type eef_inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --world_cost_success 12000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/redobject_straight_push --demo_type robot_demo  --robot_cost_weight 50000 --robot_cost_success 0.02 --red_robot True >eefnoinpaint_robotdemo_envcem.out  2>&1 &
wait
