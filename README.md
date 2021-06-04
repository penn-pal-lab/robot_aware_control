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
* `locobot_rospkg` contains the ROS node for running the WidowX robot.
* `robonet` contains the RoboNet codebase, which we use for extracting the subset of RoboNet robots for pre-training.

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

### Vanilla Model
Train on RoboNet data
```
python -m src.prediction.trainer --jobname robonet_vanilla_rawac_tile_norm --wandb True --data_root /scratch/anonymous/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment train_robonet --preprocess_action raw --train_val_split 0.95 --model_use_robot_state False --model_use_mask False --model_use_future_mask False --model_use_future_robot_state False --random_snippet True --wandb_group camera_space_tile --lstm_group_norm True
```
Finetune on WidowX data
```
python -m src.prediction.trainer --jobname ft389locobot_vanilla_rawac_145k_norm --wandb True --data_root /scratch/anonymous/Robonet --batch_size 10 --n_future 5 --n_past 1 --n_eval 10 --g_dim 256 --z_dim 64 --model svg --niter 51 --epoch_size 40 --checkpoint_interval 10 --eval_interval 2 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment finetune_locobot --preprocess_action raw --dynamics_model_ckpt logs/robonet_vanilla_rawac_tile_norm/ckpt_145500.pt --model_use_mask False --model_use_robot_state False --model_use_future_mask False --model_use_future_robot_state False --finetune_num_train 1000 --finetune_num_test 400 --learned_robot_model False --scheduled_sampling_k 550 --robot_joint_dim 5 --wandb_group ft_camera_space_tile --lstm_group_norm True
```

### Robot-Aware Model

Train on RoboNet data
```
python -m src.prediction.trainer --jobname robonet_roboaware_rawstateac_tile_norm --wandb True --data_root /scratch/anonymous/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss dontcare_l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment train_robonet --preprocess_action raw --train_val_split 0.95 --model_use_robot_state True --model_use_mask True --model_use_future_mask True --model_use_future_robot_state True --random_snippet True --wandb_group camera_space_tile --lstm_group_norm True
```

Finetune on WidowX200 data
```
python -m src.prediction.trainer --jobname ft389locobot_roboaware_rawstateac_144k_norm --wandb True --data_root /scratch/anonymous/Robonet --batch_size 10 --n_future 5 --n_past 1 --n_eval 10 --g_dim 256 --z_dim 64 --model svg --niter 51 --epoch_size 40 --checkpoint_interval 10 --eval_interval 2 --reconstruction_loss dontcare_l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment finetune_locobot --preprocess_action raw --dynamics_model_ckpt logs/robonet_roboaware_rawstateac_tile_norm/ckpt_144000.pt --model_use_mask True --model_use_robot_state True --model_use_future_mask True --model_use_future_robot_state True --finetune_num_train 1000 --finetune_num_test 400 --learned_robot_model False --scheduled_sampling_k 550 --robot_joint_dim 5 --wandb_group ft_camera_space_tile --lstm_group_norm True
```

## Visual MPC Experiments

### Using LoCoBot Vanilla Model

Finetuned vanilla model

```bash
python -m locobot_rospkg.nodes.visual_MPC_controller --g_dim 256 --z_dim 64 --model svg --last_frame_skip True --lstm_group_norm True --action_dim 5 --robot_dim 5 --preprocess_action raw  --model_use_mask False --model_use_robot_state False --model_use_heatmap False --dynamics_model_ckpt checkpoints/vanilla_ckpt_10200.pt --action_candidates 300 --candidates_batch_size 300 --cem_init_std 0.03 --sparse_cost False --horizon 5 --object shark --push_type right
```

Finetuned vanilla model with Roboaware cost

```bash
python -m locobot_rospkg.nodes.visual_MPC_controller --g_dim 256 --z_dim 64 --model svg --last_frame_skip True --lstm_group_norm True --action_dim 5 --robot_dim 5 --preprocess_action raw  --model_use_mask False --model_use_robot_state False --model_use_heatmap False --dynamics_model_ckpt checkpoints/vanilla_ckpt_10200.pt --action_candidates 300 --candidates_batch_size 300 --cem_init_std 0.03 --sparse_cost False --horizon 5 --reward_type dontcare --object watermelon --push_type right
```

Full vanilla model

```bash
python -m locobot_rospkg.nodes.visual_MPC_controller --g_dim 256 --z_dim 64 --model svg --last_frame_skip True --action_dim 5 --robot_dim 5 --preprocess_action raw  --model_use_mask False --model_use_robot_state False --model_use_heatmap False --dynamics_model_ckpt checkpoints/locobot_689_tile_ckpt_136500.pt --action_candidates 300 --candidates_batch_size 300 --cem_init_std 0.015 --sparse_cost True
```

Full vanilla model with Roboaware cost

```bash
python -m locobot_rospkg.nodes.visual_MPC_controller --g_dim 256 --z_dim 64 --model svg --last_frame_skip True --action_dim 5 --robot_dim 5 --preprocess_action raw  --model_use_mask False --model_use_robot_state False --model_use_heatmap False --dynamics_model_ckpt checkpoints/locobot_689_tile_ckpt_136500.pt --action_candidates 300 --candidates_batch_size 300 --cem_init_std 0.03 --sparse_cost False --horizon 5 --reward_type dontcare --object shark --push_type right
```

### Using LoCoBot Roboaware Model

```bash
python -m locobot_rospkg.nodes.visual_MPC_controller --g_dim 256 --z_dim 64 --model svg --last_frame_skip True --action_dim 5 --robot_dim 5 --preprocess_action raw  --model_use_mask True --model_use_robot_state True --model_use_future_mask True --model_use_future_robot_state True --lstm_group_norm True --robot_joint_dim 5 --dynamics_model_ckpt checkpoints/roboaware_ckpt_10200.pt --reconstruction_loss dontcare_l1 --reward_type dontcare --action_candidates 300 --candidates_batch_size 300 --cem_init_std 0.03 --sparse_cost False --horizon 5 --object shark --push_type right
```