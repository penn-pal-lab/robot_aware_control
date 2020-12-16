#!/usr/bin/env bash
#SBATCH --cpus-per-task 8
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=1std_30kdet_cem

python -um src.mbrl.episode_runner --wandb True --jobname 2time_inpaintdemo_30kenvcem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob True  --use_env_dynamics True --num_episodes 100 --most_recent_background False  --subgoal_threshold 4000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,1 --object_demo_dir demos/straight_push --stoch False  --demo_type object_inpaint_demo --cem_init_std 1 >2time_inpaintdemo_30kenvcem.out 2>&1 &

python -um src.mbrl.episode_runner --wandb True --jobname 2time_inpaintdemo_30kmodelcem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200  --topk 5   --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob True  --use_env_dynamics False --num_episodes 100 --most_recent_background False  --subgoal_threshold 4000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,1 --object_demo_dir demos/straight_push --dynamics_model_ckpt logs/30kdata_skip_inpaint_ssl2mask_det/ckpt_112500.pt --stoch False  --debug_cem True --demo_type object_inpaint_demo --cem_init_std 1 >2time_inpaintdemo_30kmodelcem.out 2>&1 &
wait
