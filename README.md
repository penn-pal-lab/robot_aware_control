## Know Thyself: Transferable Visual Control Policies Through Robot-Awareness 

#### [[Project Website]](https://edwardshu.com/rac) [[ICLR Talk]](https://iclr.cc/virtual/2022/poster/6041)

[Edward S. Hu](https://edwardshu.com/), [Kun Huang](https://www.linkedin.com/in/kun-huang-620034171/), [Oleh Rybkin](https://www.seas.upenn.edu/~oleh/), [Dinesh Jayaraman](https://www.seas.upenn.edu/~dineshj/)



<a href="https://edwardshu.com/rac">
<p align="center">
<img src="https://edwardshu.com/rac/img/wide_teaser.jpg" width="600">
</p>
</img></a>

Training visual control policies from scratch on a new robot typically requires generating large amounts of robot-specific data. How might we leverage data previously collected on another robot to reduce or even completely remove this need for robot-specific data? We propose a "robot-aware control" paradigm that achieves this by exploiting readily available knowledge about the robot. We then instantiate this in a robot-aware model-based RL policy by training modular dynamics models that couple a transferable, robot-aware world dynamics module with a robot-specific, potentially analytical, robot dynamics module. This also enables us to set up visual planning costs that separately consider the robot agent and the world. Our experiments on tabletop manipulation tasks with simulated and real robots demonstrate that these plug-in improvements dramatically boost the transferability of visual model-based RL policies, even permitting zero-shot transfer of visual manipulation skills onto new robots. 

If you find this work useful in your research, please cite:

```
@inproceedings{
  hu2022know,
  title={Know Thyself: Transferable Visual Control Policies Through Robot-Awareness},
  author={Edward S. Hu and Kun Huang and Oleh Rybkin and Dinesh Jayaraman},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=o0ehFykKVtr}
}
```

## Note to prospective users
This codebase is intended to be a reference for researchers. It is not production ready! For example, there are a lot of hard-coded values in the camera calibration / mask annotation part of the pipline. Please contact the authors for any questions.

## Codebase structure

The codebase is structured as follows:

* `scripts` contains all SLURM scripts for running jobs on cluster
* `robonet` contains Robonet data loading and annotation code
* `locobot_rospkg` contains ROS code for running real-world MBRL robotics experiments.
* `src` contains all python code for training the video prediction model and planning actions
    * `cem` contains the CEM policy for generating actions with a model
    * `config` contains all hyperparameter and configuration variables for algorithms, environments, etc.
    * `cyclegan` contains the CycleGAN domain transfer baseline
    * `datasets` contains the dataloading and data generation code.
    * `env` contains all simulated environments. 
    * `mbrl` contains the simulated MBRL experiments.
    * `prediction` contains all predictive model training code. The model is a SVG video prediction model.
    * `utils` contains some logging and visualization code
    * `visualizations` contains more plotting and visualization code.

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

## Running Experiments

### Datasets
RAC uses the [Robonet](https://www.robonet.wiki/) dataset as well as several datasets collected in the lab. We annotated a subset of Robonet with the robot masks, and that subset is used for training the robot-aware model. Contact the authors for more details.

### Training Video Prediction

To train the vanilla SVG prediction:
```bash
COST="l1"
ACTION="raw"
NAME="sawyerallvp_vanilla3_${ACTION}_convsvg_l1"

python -um src.prediction.multirobot_trainer  --jobname $NAME --wandb True --data_root /scratch/edward/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 512 --z_dim 64 --model svg --niter 1000 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5  --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 5 --lr 0.0001 --experiment singlerobot --preprocess_action raw --world_error_dict widowx1_c0_world_error.pkl --train_val_split 0.95 --model_use_robot_state False --model_use_mask False --random_snippet True >"${NAME}.out" 2>&1 &
```

```bash
COST="dontcare_l1"
ACTION="raw"
NAME="sawyerallvp_norobot3_${ACTION}_svg_l1"

python -um src.prediction.multirobot_trainer  --jobname $NAME --wandb True --data_root /scratch/edward/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 512 --z_dim 64 --model svg --niter 1000 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss $COST --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 5 --lr 0.0001 --experiment singlerobot --preprocess_action raw --world_error_dict widowx1_c0_world_error.pkl --train_val_split 0.95 --model_use_robot_state True --model_use_mask True --model_use_future_mask True --random_snippet True >"${NAME}.out" 2>&1 &
```
### Visual MPC Experiments
We provide all the ROS control code in `locobot_rospkg`.
Contact the authors for more details.

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
