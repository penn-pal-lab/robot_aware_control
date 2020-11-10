# Robot aware cost and dynamics

# Installation

## Prerequisites
* Ubuntu 18.04 or macOS
* Python 3.6 or above
* Mujoco 2.0

## MuJoCo Installation
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

## Python Installation
1. Install python dependencies
```bash
# Run the rest for both Ubuntu and macOS
$ pip install -r requirements.txt
```

## Testing the codebase
Try running this command.
```py
 python -m src.mbrl.cem.cem --wandb False --jobname modelcem --multiview True --img_dim 64 --reward_type inpaint --record_trajectory False --action_candidates 100 --opt_iter 3 --horizon 4 --push_dist 0.1 --max_episode_length 10  --large_block True --norobot_pixels_ob True   --debug_cem False --topk 3 --blur_sigma 2 --use_env_dynamics True
```
It should start running the CEM policy on a block pushing environment. Check your `logs/modelcem/video` folder to see each episode's mp4 output.


## Troubleshooting

Ran into C++ compilation error for `mujoco-py`. Had to correctly symlink GCC-6 to gcc
command for it to compile since it was using the system gcc, which was gcc-11.

Getting mujoco GPU rendering on the slurm cluster is super annoying. Got the error
GLEW initialization error: Missing GL version.

https://github.com/openai/mujoco-py/issues/408

1. Need to root login to the specific node you want to do MuJoCo GPU rendering on, and then
create a fake /usr/lib/nvidia-000 folder for MuJoCo to detect.
2. Make sure your LD_PRELOAD contians libGL and libGLEW links