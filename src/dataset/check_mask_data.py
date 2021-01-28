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

def generate_baxter_data():
    """
    Generate for left and right arm viewpoints
    """
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [120, 160]
    hparams.cams_to_load = [0]

    df = pd.read_pickle(metadata_path, compression="gzip")
    baxter_df = df.loc["baxter" == df["robot"]]
    left = ["left" in x for x in baxter_df.index]
    right = [not x for x in left]

    # generate masks for right arm
    arm = "right"
    baxter_subset = baxter_df[left if arm == "left" else right]
    camera_extrinsics = np.array(
        [
            [0.59474902, -0.48560866, 0.64066983, 0.00593267],
            [-0.80250365, -0.40577623, 0.4374169, -0.84046503],
            [0.04755516, -0.77429315, -0.63103774, 0.45875102],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    env = BaxterMaskEnv()
    env.arm = arm
    env.set_opencv_camera_pose("main_cam", camera_extrinsics)
    hdf5_list = baxter_subset.index
    generate_robot_masks(env, baxter_df, hdf5_list, hparams)

    hdf5_list = baxter_subset.sample(5).index
    check_robot_masks(baxter_df, hdf5_list, hparams)


    # generate masks for left arm
    arm = "left"
    baxter_subset = baxter_df[left if arm == "left" else right]
    camera_extrinsics = np.array(
            [
                [0.05010049, 0.5098481, -0.85880432, 1.70268951],
                [0.99850135, -0.00660876, 0.05432662, 0.26953027],
                [0.02202269, -0.86023906, -0.50941512, 0.48536055],
                [0.0, 0.0, 0.0, 1.0],
            ]
    )

    env = BaxterMaskEnv()
    env.arm = arm
    env.set_opencv_camera_pose("main_cam", camera_extrinsics)
    hdf5_list = baxter_subset.index
    generate_robot_masks(env, baxter_df, hdf5_list, hparams)

    hdf5_list = baxter_subset.sample(5).index
    check_robot_masks(baxter_df, hdf5_list, hparams)

def generate_sawyer_data():
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [120, 160]
    hparams.cams_to_load = [0]

    df = pd.read_pickle(metadata_path, compression="gzip")
    sawyer_df = df.loc["sawyer" == df["robot"]]
    sawyer_subset = sawyer_df["sudri0" == sawyer_df["camera_configuration"]]

    camera_extrinsics = np.array(
        [
            [-0.01290487, 0.62117762, -0.78356355, 1.21061856],
            [1, 0.00660994, -0.01122798, 0.01680913],
            [-0.00179526, -0.78364193, -0.62121019, 0.47401633],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    env = SawyerMaskEnv()
    env.set_opencv_camera_pose("main_cam", camera_extrinsics)
    hdf5_list = sawyer_subset.index
    generate_robot_masks(env, sawyer_df, hdf5_list, hparams)

    hdf5_list = sawyer_subset.sample(5).index
    check_robot_masks(sawyer_df, hdf5_list, hparams)

def generate_widowx_data():
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [120, 160]
    hparams.cams_to_load = [0]

    df = pd.read_pickle(metadata_path, compression="gzip")
    widowx_df = df.loc["widowx" == df["robot"]]
    widowx_subset = widowx_df[widowx_df["camera_configuration"] == "widowx1"]

    camera_extrinsics = np.array(
        [
            [-0.17251765, 0.5984481, -0.78236663, 0.37869496],
            [-0.98499368, -0.10885336, 0.13393427, -0.04712975],
            [-0.00501052, 0.79373221, 0.60824672, 0.15596613],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    env = WidowXMaskEnv()
    env.set_opencv_camera_pose("main_cam", camera_extrinsics)
    hdf5_list = widowx_subset.index
    generate_robot_masks(env, widowx_df, hdf5_list, hparams)

    hdf5_list = widowx_subset.sample(10).index
    check_robot_masks(widowx_df, hdf5_list, hparams)


def check_baxter():
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [120, 160]
    hparams.cams_to_load = [0]
    df = pd.read_pickle(metadata_path, compression="gzip")
    baxter_df = df.loc["baxter" == df["robot"]]
    left = ["left" in x for x in baxter_df.index]
    right = [not x for x in left]

    # generate masks for right arm
    arm = "right"
    baxter_subset = baxter_df[left if arm == "left" else right]
    hdf5_list = baxter_subset.index
    # check_gripper_state(baxter_subset, hdf5_list, hparams)
    check_actions(baxter_subset, hdf5_list, hparams)

    # generate masks for left arm
    arm = "left"
    baxter_subset = baxter_df[left if arm == "left" else right]
    hdf5_list = baxter_subset.index
    # check_gripper_state(baxter_subset, hdf5_list, hparams)
    check_actions(baxter_subset, hdf5_list, hparams)

def check_sawyer():
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [120, 160]
    hparams.cams_to_load = [0]

    df = pd.read_pickle(metadata_path, compression="gzip")
    sawyer_df = df.loc["sawyer" == df["robot"]]
    sawyer_subset = sawyer_df["sudri0" == sawyer_df["camera_configuration"]]
    hdf5_list = sawyer_subset.index
    # check_gripper_state(sawyer_subset, hdf5_list, hparams)
    check_actions(sawyer_subset, hdf5_list, hparams)


def check_widowx():
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [120, 160]
    hparams.cams_to_load = [0]

    df = pd.read_pickle(metadata_path, compression="gzip")
    widowx_df = df.loc["widowx" == df["robot"]]
    widowx_subset = widowx_df[widowx_df["camera_configuration"] == "widowx1"]
    hdf5_list = widowx_subset.index
    # check_gripper_state(widowx_subset, hdf5_list, hparams)
    check_actions(widowx_subset, hdf5_list, hparams)

def check_gripper_state(df, hdf5_list, hparams):
    """
    Checks to see if gripper states are normalized correctly
    """
    overall_gripper_max = 0
    overall_gripper_min = 10
    overall_eef_max = [0,0, 0]
    overall_eef_min = [10, 10, 10]
    for traj_name in tqdm(hdf5_list, f"Checking state bounds"):
        file_metadata = df.loc[traj_name]
        ws_min = file_metadata["low_bound"]
        ws_max = file_metadata["high_bound"]
        f_name = os.path.join(robonet_root, traj_name)

        with h5py.File(f_name, "r") as hf:
            start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
            states = load_states(hf, file_metadata, hparams)[start_time:start_time + n_states]
            # only normalize the gripper
            states[:, -1] -= ws_min[-1]
            states[:, -1] /= (ws_max[-1] - ws_min[-1])
            # states *= (ws_max - ws_min)
            # states += ws_min
            gripper = states[:, -1]
            eef_pos = states[:, :3]
            # record the max possible eef_pos
            eef_max = eef_pos.max(0)
            eef_min = eef_pos.min(0)
            overall_eef_max = np.maximum(eef_max, overall_eef_max)
            overall_eef_min = np.minimum(eef_min, overall_eef_min)

            gripper_max = gripper.max(0)
            gripper_min = gripper.min(0)
            overall_gripper_max = np.maximum(gripper_max, overall_gripper_max)
            overall_gripper_min = np.minimum(gripper_min, overall_gripper_min)

            # if not ((eef_pos >= -0.05).all() and (eef_pos <= 1.05).all()):
            #     print("eef_pos out of bounds")
            #     ipdb.set_trace()
            # if not ((gripper >= -1.05).all() and (gripper <= 1.05).all()):
            #     print("gripper_pos out of bounds")
            #     ipdb.set_trace()
    print("eef bounds")
    print("min", overall_eef_min)
    print("max", overall_eef_max)

    print("gripper bounds")
    print("min", overall_gripper_min)
    print("max", overall_gripper_max)

def check_actions(df, hdf5_list, hparams):
    """
    Checks to see if action bounds are correct
    """
    overall_gripper_max = 0
    overall_gripper_min = 10
    overall_eef_max = [0,0, 0]
    overall_eef_min = [10, 10, 10]
    overall_eef_sum = np.zeros(3)
    for traj_name in tqdm(hdf5_list, f"Checking action bounds"):
        file_metadata = df.loc[traj_name]
        # ws_min = file_metadata["low_bound"]
        # ws_max = file_metadata["high_bound"]
        f_name = os.path.join(robonet_root, traj_name)

        with h5py.File(f_name, "r") as hf:
            start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
            actions = load_actions(hf, file_metadata, hparams)[start_time:start_time + n_states]
            gripper = actions[:, -1]
            eef_pos = actions[:, :3]
            # record the max possible eef_pos
            eef_max = eef_pos.max(0)
            eef_min = eef_pos.min(0)
            overall_eef_max = np.maximum(eef_max, overall_eef_max)
            overall_eef_min = np.minimum(eef_min, overall_eef_min)
            overall_eef_sum += np.abs(eef_pos).mean(0)

            gripper_max = gripper.max(0)
            gripper_min = gripper.min(0)
            overall_gripper_max = np.maximum(gripper_max, overall_gripper_max)
            overall_gripper_min = np.minimum(gripper_min, overall_gripper_min)

            # if not ((eef_pos >= -0.05).all() and (eef_pos <= 1.05).all()):
            #     print("eef_pos out of bounds")
            #     ipdb.set_trace()
            # if not ((gripper >= -1.05).all() and (gripper <= 1.05).all()):
            #     print("gripper_pos out of bounds")
            #     ipdb.set_trace()
    print("eef bounds")
    print("min", overall_eef_min)
    print("max", overall_eef_max)

    print("gripper bounds")
    print("min", overall_gripper_min)
    print("max", overall_gripper_max)

    print("eef average magnitude", overall_eef_sum / len(hdf5_list))


def check_robot_masks(df, hdf5_list, hparams):
    """
    Generates mask gifs to check accuracy
    """
    robot = df["robot"][0]
    for traj_name in tqdm(hdf5_list, f"{robot} masks"):
        f_metadata = df.loc[traj_name]
        f_name = os.path.join(robonet_root, traj_name)
        start_time, n_states = 0, min([f_metadata['state_T'], f_metadata['img_T'], f_metadata['action_T'] + 1])

        with h5py.File(f_name, "r") as f:
            # TODO: handle for viewpoint
            masks = f["env/cam0_mask"][:]
            imgs = load_camera_imgs(0, f, f_metadata, hparams.img_size, start_time, n_states)
        imgs[masks] = (0, 255, 255)
        imageio.mimwrite(f"{traj_name[:-5]}.gif", imgs)

def generate_robot_masks(env, df, hdf5_list, hparams):
    robot = df["robot"][0]
    for traj_name in tqdm(hdf5_list, f"{robot} masks"):
        f_metadata = df.loc[traj_name]
        f_name = os.path.join(robonet_root, traj_name)
        # save the masks back into the h5py file
        with h5py.File(f_name, "r") as f:
            qposes = load_qpos(f, f_metadata, hparams)
        masks = env.generate_masks(qposes)
        with h5py.File(f_name, "a") as f:
            # TODO create dataset given the viewpoint instead of assuming 0
            if "cam0_mask" in f["env"]:
                del f["env/cam0_mask"]
            f["env"].create_dataset("cam0_mask", data=masks, compression="gzip")


if __name__ == "__main__":
    """
    Generates masks for each robot, and stores it in the target directory.
    """
    num_videos = {
        "baxter": 100,
        "sawyer": 100,
        "widowx": 100,
    }
    rng = 123
    # generate_sawyer_data()
    # generate_baxter_data()
    generate_widowx_data()

    # print("Checking Sawyer")
    # check_sawyer()
    # print()

    # print("Checking WidowX")
    # check_widowx()
    # print()

    # print("Checking Baxter")
    # check_baxter()
