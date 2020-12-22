#!/usr/bin/env bash
#SBATCH --cpus-per-task 8
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=wrongcolor_noinpaint_objonly

python -um src.mbrl.episode_runner --wandb False --jobname noinpaint_wrongcolor_robdemo_envcem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --subgoal_threshold 15000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/realvp_straight_push  --stoch False  --demo_type robot_demo --cem_init_std 1 >noinpaint_realvp_wrongcolor_robdemo_envcem.out 2>&1 &

python -um src.mbrl.episode_runner --wandb False --jobname noinpaint_wrongcolor_robdemo_modelcem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics False --num_episodes 100 --most_recent_background False  --subgoal_threshold 15000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/realvp_straight_push --dynamics_model_ckpt logs/realvp_randomdata_noinpaint_det/ckpt_128125.pt --stoch False  --debug_cem True --demo_type robot_demo --cem_init_std 1 >noinpaint_realvp_wrongcolor_robdemo_modelcem.out 2>&1 &
wait
