#!/usr/bin/env bash
#SBATCH --cpus-per-task 8
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=1std_30kdet_cem

#python -um src.mbrl.episode_runner --wandb False --jobname noinpaint_randomdataenvcem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --subgoal_threshold 8500 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,1 --object_demo_dir demos/straight_push --stoch False  --demo_type object_inpaint_demo --cem_init_std 1 >noinpaint_inpaintdemo_1std30kenvcem.out 2>&1 &

python -um src.mbrl.episode_runner --wandb False --jobname noinpaint_randomdata_modelcem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob False  --use_env_dynamics False --num_episodes 100 --most_recent_background False  --subgoal_threshold 8000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,1 --object_demo_dir demos/straight_push --dynamics_model_ckpt logs/randomdata_noinpaint_det/ckpt_81250.pt --stoch False  --debug_cem True --demo_type robot_demo --cem_init_std 1 >noinpaint_randomdata_modelcem.out 2>&1 &
wait
