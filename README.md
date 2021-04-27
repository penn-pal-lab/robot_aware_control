# Robot aware cost and dynamics

This project investigates how we can use our knowledge of the robot in the scene to improve pixel cost functions and pixel dynamics prediction.

## Codebase structure

The codebase is structured as follows:

* `scripts` contains all SLURM scripts for running jobs on cluster
* `src` contains all source code.
    * `config` contains all hyperparameter and configuration variables for algorithms, environments, etc.
    * `env` contains all environments. We mainly use the `fetch` environment, which features a Fetch 7DOF robot with EEF positional control on a tabletop workspace.
    * `datasets` contains the dataloading and data generation code.
    * `cem` contains the CEM policy for generating actions with a model
    * `mbrl` contains the policy evaluation code.
    * `prediction` contains all predictive model training code. The model is a SVG video prediction model.
    * `utils` contains some plotting and visualization code.

## Installation

### Prerequisites

* Ubuntu 18.04 or macOS
* Python 3.6 or above
* Mujoco 2.0

### MuJoCo Installation

1. Create the `~/.mujoco` folder in your home directory. Add your MuJoCo license `mjkey.txt` into the mujoco folder. If you do not have a license key, ask one of the lab members for it.

2. Install mujoco 2.0 and add the following environment variables into `~/.bashrc` or `~/.zshrc`

```bash
# download mujoco 2.0 from https://www.roboti.us/index.html
$ wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip
$ unzip mujoco.zip -d ~/.mujoco
$ mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200

# copy mujoco license key `mjkey.txt` to `~/.mujoco`

# add mujoco to LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

# for GPU rendering (replace 418 with your nvidia driver version)
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-418
```

### Python Installation

1. Install python dependencies. This requirements file is out of date, so you'll probably
run into import errors and have to install the missing packages. Sorry!

```bash
# Run the rest for both Ubuntu and macOS
$ pip install -r requirements.txt
```

## Running the Code

Here, we will generate some demonstrations, and then run CEM to follow the demonstrations.

### Generating demonstrations

For the clutter environment, we will generate block pushing demonstrations.

```bash
python -m src.dataset.collect_clutter_data
```

This will generate 100 block pushing demonstrations saved into `demos/straight_push`. You can change the number of demonstrations, inpainting type, etc. in the file.

### Running Demonstration Following Episodes

```bash
python -m src.mbrl.episode_runner --wandb False --jobname democem --multiview True --img_dim 64 --reward_type inpaint  --action_candidates 200 --topk 10  --opt_iter 2 --horizon 2  --max_episode_length 10  --norobot_pixels_ob True  --use_env_dynamics True --num_episodes 100 --most_recent_background False --action_repeat 1 --subgoal_threshold 5000 --sequential_subgoal True --demo_cost True --subgoal_start 1 --demo_timescale 2 --camera_ids 0,1 --object_demo_dir demos/straight_push
```

Once we have the demonstrations, we can load them and have the CEM policy attempt to follow them.

## Troubleshooting

Ran into C++ compilation error for `mujoco-py`. Had to correctly symlink GCC-6 to gcc
command for it to compile since it was using the system gcc, which was gcc-11.

Getting mujoco GPU rendering on the slurm cluster is super annoying. Got the error
GLEW initialization error: Missing GL version.

https://github.com/openai/mujoco-py/issues/408

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

1. Need to root login to the specific node you want to do MuJoCo GPU rendering on, and then
create a fake /usr/lib/nvidia-000 folder for MuJoCo to detect.
2. Make sure your LD_PRELOAD contians libGL and libGLEW links

## Training Video Prediction

### LoCoBot Vanilla Model

```bash
python -m src.prediction.multirobot_trainer --jobname locobot --wandb False --data_root /mnt/ssd1/pallab/locobot_data --batch_size 10 --n_future 5 --n_past 1 --n_eval 10 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --checkpoint_interval 100 --eval_interval 5 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment train_locobot_singleview --preprocess_action raw --random_snippet True --model_use_mask False --model_use_robot_state False --model_use_heatmap False

CUDA_VISIBLE_DEVICES=0 python -m src.prediction.multirobot_trainer --jobname locobot_1000 --wandb True --data_root /home/huangkun/locobot_data --batch_size 10 --n_future 5 --n_past 1 --n_eval 10 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --checkpoint_interval 100 --eval_interval 5 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment train_locobot_singleview --preprocess_action raw --random_snippet True --model_use_mask False --model_use_robot_state False --model_use_heatmap False
```

## Visual MPC Experiments

### Using LoCoBot Vanilla Model

Finetuned vanilla model

```bash
python -m locobot_rospkg.nodes.visual_MPC_controller --g_dim 256 --z_dim 64 --model svg --last_frame_skip True --lstm_group_norm True --action_dim 5 --robot_dim 5 --preprocess_action raw  --model_use_mask False --model_use_robot_state False --model_use_heatmap False --dynamics_model_ckpt checkpoints/vanilla_ckpt_10200.pt --action_candidates 300 --candidates_batch_size 300 --cem_init_std 0.015 --sparse_cost True
```

Full vanilla model

```bash
python -m locobot_rospkg.nodes.visual_MPC_controller --g_dim 256 --z_dim 64 --model svg --last_frame_skip True --action_dim 5 --robot_dim 5 --preprocess_action raw  --model_use_mask False --model_use_robot_state False --model_use_heatmap False --dynamics_model_ckpt checkpoints/locobot_689_tile_ckpt_136500.pt --action_candidates 300 --candidates_batch_size 300 --cem_init_std 0.015 --sparse_cost True
```

### Using LoCoBot Roboaware Model

```bash
python -m locobot_rospkg.nodes.visual_MPC_controller --g_dim 256 --z_dim 64 --model svg --last_frame_skip True --action_dim 5 --robot_dim 5 --preprocess_action raw  --model_use_mask True --model_use_robot_state True --model_use_future_mask True --model_use_future_robot_state True --lstm_group_norm True --robot_joint_dim 5 --dynamics_model_ckpt checkpoints/roboaware_ckpt_10200.pt --reconstruction_loss dontcare_l1 --reward_type dontcare --action_candidates 300 --candidates_batch_size 300 --cem_init_std 0.015 --sparse_cost True
```
