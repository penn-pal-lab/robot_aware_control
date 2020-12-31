#!/usr/bin/env bash
#SBATCH --cpus-per-task 8
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=inpaint_realisticmask_modelcem

python -um src.mbrl.episode_runner --wandb False --jobname inpaint_realisticmask_modelcem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200  --topk 5 --opt_iter 2 --horizon 2  --max_episode_length 13  --norobot_pixels_ob True --use_env_dynamics False --num_episodes 100 --most_recent_background False  --subgoal_threshold 15000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,2 --object_demo_dir demos/realisticrobotmask_straight_push --dynamics_model_ckpt logs/realisticmask_inpaint_det/ckpt_112500.pt --stoch False  --debug_cem False --demo_type object_only_demo --cem_init_std 1 --gpu 0 >inpaint_realisticmask_modelcem.out 2>&1 &
wait
