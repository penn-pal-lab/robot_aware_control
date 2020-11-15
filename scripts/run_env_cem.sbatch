#!/usr/bin/env bash
#SBATCH --cpus-per-task 12
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=dinesh-compute
#SBATCH --qos=dinesh-high
#SBATCH --job-name=democem

# baseline with robot pixels
# python -um src.mbrl.cem.demo_cem --wandb False --jobname robot_democem2 --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200 --topk 10  --opt_iter 2 --horizon 2  --max_episode_length 10  --norobot_pixels_ob False  --use_env_dynamics True --num_episodes 100 --most_recent_background False --action_repeat 1 --subgoal_threshold 10000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,1 --object_demo_dir demos/straight_push >robot_democem.out 2>&1  &

# static inpaint
# python -um src.mbrl.cem.demo_cem --wandb False --jobname mr_inpaint_democem2 --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200 --topk 10  --opt_iter 2 --horizon 2  --max_episode_length 10  --norobot_pixels_ob True  --use_env_dynamics True --num_episodes 100 --most_recent_background True --action_repeat 1 --subgoal_threshold 5000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,1 --object_demo_dir demos/straight_push >mr_inpaint_democem.out 2>&1  &
python -um src.dataset.collect_clutter_data >svg_data.out 2>&1 &
wait
