#!/usr/bin/env bash
#SBATCH --cpus-per-task 4
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=inpaintsvg_cem

# most recent inpaint SVG CEM
# python -um src.mbrl.episode_runner --wandb False --jobname learneddemocem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 800 --candidates_batch_size 400 --topk 10   --opt_iter 2 --horizon 2  --max_episode_length 10  --norobot_pixels_ob True  --use_env_dynamics False --num_episodes 100 --most_recent_background True  --subgoal_threshold 5000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,1 --object_demo_dir demos/straight_push  --dynamics_model_ckpt logs/mr_cluttersvg/ckpt_55000.pt --debug_cem True  >mrinpaintsvg_cem.out 2>&1 &

# static inpaint SVG CEM
python -um src.mbrl.episode_runner --wandb False --jobname inpaintdemocem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 800 --candidates_batch_size 400 --topk 10   --opt_iter 2 --horizon 2  --max_episode_length 10  --norobot_pixels_ob True  --use_env_dynamics False --num_episodes 100 --most_recent_background False  --subgoal_threshold 5000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,1 --object_demo_dir demos/straight_push  --dynamics_model_ckpt logs/cluttersvg2/ckpt_280000.pt --debug_cem True  >inpaintsvg_cem.out 2>&1 &
wait
