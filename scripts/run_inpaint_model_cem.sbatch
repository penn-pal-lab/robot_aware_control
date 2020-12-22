#!/usr/bin/env bash
#SBATCH --cpus-per-task 8
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=test_eef_inpaint_cost

python -um src.mbrl.episode_runner --wandb False --jobname realvp_inpaintdemo_envcem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob True  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --subgoal_threshold 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/realvp_straight_push --stoch False  --debug_cem False --demo_type object_only_demo >realvp_inpaintdemo_envcem.out 2>&1 &

python -um src.mbrl.episode_runner --wandb False --jobname realvp_0robotweight_inpaintdemo_envcem --multiview True --img_dim 64 --reward_type eef_inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob True  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --subgoal_threshold 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/realvp_straight_push --stoch False  --debug_cem False --demo_type object_only_demo  --robot_weight 0 --subgoal_robot_threshold 100000 >realvp_0robotweight_inpaintdemo_envcem.out 2>&1 &
wait
