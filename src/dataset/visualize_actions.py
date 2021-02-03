#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from warnings import simplefilter # disable tensorflow warnings
simplefilter(action='ignore', category=FutureWarning)

import h5py
import imageio
import ipdb
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt

import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)


from robonet.robonet.datasets.util.hdf5_loader import (default_loader_hparams,
                                                       load_actions,
                                                       load_camera_imgs,
                                                       load_qpos, load_states)
from robonet.robonet.datasets.util.metadata_helper import load_metadata
from src.env.robotics.masks.baxter_mask_env import BaxterMaskEnv
from src.env.robotics.masks.sawyer_mask_env import SawyerMaskEnv
from src.env.robotics.masks.widowx_mask_env import WidowXMaskEnv
from tqdm import tqdm

robonet_root = "/media/ed/hdd/Datasets/Robonet/hdf5/"
metadata_path = "/media/ed/hdd/Datasets/Robonet/hdf5/meta_data.pkl"

def load_all_states(df, hparams, hdf5_list):
    all_states = []
    prev_ws_min = prev_ws_max = None

    for traj_name in tqdm(hdf5_list, f"Collecting states"):
        file_metadata = df.loc[traj_name]
        ws_min = file_metadata["low_bound"]
        ws_max = file_metadata["high_bound"]
        if prev_ws_min is None:
            prev_ws_min = ws_min
            prev_ws_max = ws_max
        ws_min_close = np.allclose(prev_ws_min, ws_min)
        ws_max_close = np.allclose(prev_ws_max, ws_max)
        if not (ws_min_close and ws_max_close):
            print(traj_name, "ws not same")
            print(ws_min, prev_ws_min)
            print(ws_max, prev_ws_max)
        prev_ws_min = ws_min
        prev_ws_max = ws_max


        f_name = os.path.join(robonet_root, traj_name)

        with h5py.File(f_name, "r") as hf:
            start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
            states = load_states(hf, file_metadata, hparams)
            assert states.shape[-1] == 5
            all_states.append(states)
    return np.asarray(all_states), ws_min, ws_max

def load_all_actions(df, hparams, hdf5_list):
    all_actions = []

    for traj_name in tqdm(hdf5_list, f"Collecting actions"):
        file_metadata = df.loc[traj_name]
        f_name = os.path.join(robonet_root, traj_name)

        with h5py.File(f_name, "r") as hf:
            actions = load_actions(hf, file_metadata, hparams)
            assert actions.shape[-1] == 4, actions.shape
            all_actions.append(actions)
    return np.asarray(all_actions)

def plot_actions(robot_name, robot_data, robot_low, robot_high, action_data):
    errors = []
    for i in range(len(robot_data[robot_name])):
        states = robot_data[robot_name][i]
        low = robot_low[robot_name][:3]
        high = robot_high[robot_name][:3]
        states = states[:, :4].copy()
        states[:, :3] = (states[:, :3] * (high - low)) + low

        actions = action_data[robot_name][i]
        for t, ac in enumerate(actions):
            state = states[t]
            next_state = states[t+1]
            true_state_diff = next_state - state
            # truncate action by boundary before computing error
            pred_state = state[:3] + ac[:3]
            trunc_pred_state = np.clip(pred_state, low, high)
            trunc_action = trunc_pred_state - state[:3]
            error = true_state_diff[:3] - trunc_action[:3]
            errors.append(error[:3])

    errors = np.asarray(errors)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(errors[:, 0], errors[:, 1], errors[:, 2])
    ax.set_box_aspect([1,1,1])
    plt.show()

def load_data(df, hparams):

    widowx_df = df.loc["widowx" == df["robot"]]
    widowx_subset = widowx_df[widowx_df["camera_configuration"] == "widowx1"].sample(100)
    widowx_states, widowx_min, widowx_max = load_all_states(df, hparams,  widowx_subset.index)

    sawyer_df = df.loc["sawyer" == df["robot"]]
    sawyer_subset = sawyer_df["sudri0" == sawyer_df["camera_configuration"]].sample(100)
    sawyer_states, sawyer_min, sawyer_max = load_all_states(df, hparams, sawyer_subset.index)

    baxter_df = df.loc["baxter" == df["robot"]]
    left = ["left" in x for x in baxter_df.index]
    right = [not x for x in left]
    baxter_left_subset = baxter_df[left].sample(100)
    baxter_right_subset = baxter_df[right].sample(100)
    baxter_left_states, baxter_left_min, baxter_left_max = load_all_states(df,hparams,  baxter_left_subset.index)
    baxter_right_states, baxter_right_min, baxter_right_max = load_all_states(df, hparams,  baxter_right_subset.index)



    robot_data = {"baxter_left": baxter_left_states, "baxter_right": baxter_right_states, "sawyer": sawyer_states, "widowx": widowx_states}
    robot_min = {"baxter_left": baxter_left_min, "baxter_right": baxter_right_min, "sawyer": sawyer_min, "widowx": widowx_min}
    robot_max = {"baxter_left": baxter_left_max, "baxter_right": baxter_right_max, "sawyer": sawyer_max, "widowx": widowx_max}


    widowx_actions = load_all_actions(df, hparams, widowx_subset.index)
    baxter_left_actions = load_all_actions(df, hparams, baxter_left_subset.index)
    baxter_right_actions = load_all_actions(df, hparams, baxter_right_subset.index)
    sawyer_actions = load_all_actions(df, hparams, sawyer_subset.index)


    action_data = {"baxter_left": baxter_left_actions, "baxter_right": baxter_right_actions, "sawyer": sawyer_actions, "widowx": widowx_actions}

    return robot_data, robot_min, robot_max, action_data
if __name__ == "__main__":
    import pickle
    # first load widowx states
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [120, 160]
    hparams.cams_to_load = [0]

    df = pd.read_pickle(metadata_path, compression="gzip")
    # robot_data, low, high, action_data = load_data(df, hparams)
    # with open("visualize_actions.pkl", "wb") as f:
    #     pickle.dump([robot_data, low, high, action_data], f)
    with open("visualize_actions.pkl", "rb") as f:
        robot_data, low, high, action_data = pickle.load(f)
    robot_name = "sawyer"
    sawyer_df = df.loc["sawyer" == df["robot"]]
    subset = sawyer_df["sudri0" == sawyer_df["camera_configuration"]].sample(100)
    # robot_name = "widowx"
    # widowx_df = df.loc["widowx" == df["robot"]]
    # subset = widowx_df[widowx_df["camera_configuration"] == "widowx1"].sample(100)

    # robot_name="baxter_left"
    # baxter_df = df.loc["baxter" == df["robot"]]
    # left = ["left" in x for x in baxter_df.index]
    # right = [not x for x in left]
    # subset = baxter_df[left].sample(100)

    # robot_name="baxter_right"
    # subset = baxter_df[right].sample(100)
    hdf5_list = subset.index
    plot_actions( robot_name,  robot_data, low, high, action_data)
