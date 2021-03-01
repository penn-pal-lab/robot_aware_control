import os
from src.env.robotics.masks.widowx_mask_env import WidowXMaskEnv
from src.env.robotics.masks.baxter_mask_env import BaxterMaskEnv
import h5py
import imageio
import ipdb

import numpy as np
import pandas as pd
import tensorflow as tf
from robonet.robonet.datasets.util.hdf5_loader import default_loader_hparams, load_actions, load_camera_imgs, load_qpos, load_states
from src.env.robotics.masks.sawyer_mask_env import SawyerMaskEnv
from tqdm import tqdm
from src.utils.camera_calibration import world_to_camera_dict


robonet_root = "/scratch/edward/Robonet/hdf5"
metadata_path = os.path.join(robonet_root, "meta_data.pkl")

def generate_baxter_data():
    """
    Generate for left and right arm viewpoints
    """
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [64, 85]
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
    generate_robot_masks(env, baxter_df, hdf5_list, hparams, new_robonet_root)

    # hdf5_list = baxter_subset.sample(5).index
    # check_robot_masks(baxter_df, hdf5_list, hparams)


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
    generate_robot_masks(env, baxter_df, hdf5_list, hparams, new_robonet_root)

    # hdf5_list = baxter_subset.sample(5).index
    # check_robot_masks(baxter_df, hdf5_list, hparams)

def generate_sawyer_data(cam_config, cam_idx):
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [64, 85]
    hparams.cams_to_load = [cam_idx]

    new_robonet_root = f"/scratch/edward/Robonet/sawyer_views_qpos/{cam_config}_c{hparams.cams_to_load[0]}"
    os.makedirs(new_robonet_root, exist_ok=True)

    df = pd.read_pickle(metadata_path, compression="gzip")
    sawyer_df = df.loc["sawyer" == df["robot"]]
    sawyer_subset = sawyer_df[f"{cam_config}" == sawyer_df["camera_configuration"]]

    camera_extrinsics = world_to_camera_dict[f"sawyer_{cam_config}_c{hparams.cams_to_load[0]}"]
    env = SawyerMaskEnv()
    env.set_opencv_camera_pose("main_cam", camera_extrinsics)
    hdf5_list = sawyer_subset.index
    generate_robot_masks(env, sawyer_df, hdf5_list, hparams, new_robonet_root)

    # hdf5_list = sawyer_subset.sample(5).index
    # check_robot_masks(sawyer_df, hdf5_list, hparams, new_robonet_root)

def generate_all_sawyer_data():
    # cam_configs = ["sudri0", "sudri2", "vestri_table2"]
    # cams = [0,1,2]
    cam_configs = ["vestri_table2"]
    cams = [0,1,2]
    for config in cam_configs:
        for i in cams:
            generate_sawyer_data(config, i)

def generate_widowx_data():
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [64, 85]
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
    generate_robot_masks(env, widowx_df, hdf5_list, hparams, new_robonet_root)

    # hdf5_list = widowx_subset.sample(5).index
    # check_robot_masks(widowx_df, hdf5_list, hparams)




def check_robot_masks(df, hdf5_list, hparams, target_dir):
    """
    Generates mask gifs to check accuracy
    """
    robot = df["robot"][0]
    cam_idx = hparams.cams_to_load[0]
    for traj_name in tqdm(hdf5_list, f"visualizing {robot} masks"):
        traj_name_parts = traj_name.split(".")
        traj_name_parts[-2] += f"_c{cam_idx}"
        new_traj_name = ".".join(traj_name_parts)
        new_f_name = os.path.join(target_dir, new_traj_name)
        # f_name = os.path.join(new_robonet_root, traj_name)
        with h5py.File(new_f_name, "r") as f:
            masks = f["mask"][:]
            imgs = f["frames"][:]
        imgs[masks] = (0, 255, 255)
        imageio.mimwrite(f"{traj_name[:-5]}.gif", imgs)

def generate_robot_masks(env, df, hdf5_list, hparams, target_dir):
    """
    Write this into a new hdf5 file.
    Write the images, states, actions, masks
    metadata: original path, low bound, high bound, robot
    """
    target_dims = [64, 85]
    robot = df["robot"][0]
    for traj_name in tqdm(hdf5_list, f"making {robot} masks"):
        f_metadata = df.loc[traj_name]
        f_name = os.path.join(robonet_root, traj_name)
        # save the masks back into the h5py file
        with h5py.File(f_name, "r") as f:
            qposes = load_qpos(f, f_metadata, hparams)
            low_bound = f["env/low_bound"][-1]
            high_bound = f["env/high_bound"][-1]
            actions = load_actions(f, f_metadata, hparams)
            states = load_states(f, f_metadata, hparams)
            # TODO: set camera index from params
            cam_idx = hparams.cams_to_load[0]
            images = load_camera_imgs(cam_idx, f, f_metadata, target_dims)
        masks = env.generate_masks(qposes, width=target_dims[1], height=target_dims[0])
        # generate new hdf5 file
        traj_name_parts = traj_name.split(".")
        traj_name_parts[-2] += f"_c{cam_idx}"
        new_traj_name = ".".join(traj_name_parts)
        new_f_name = os.path.join(target_dir, new_traj_name)
        with h5py.File(new_f_name, "w") as f:
            # TODO create dataset given the viewpoint instead of assuming 0
            f.create_dataset("mask", data=masks, compression="gzip")
            f.attrs["cam_idx"] = cam_idx
            f.attrs["robot"] = robot
            f.attrs["traj_name"] = traj_name
            f.create_dataset("low_bound", data=low_bound, compression="gzip")
            f.create_dataset("high_bound", data=high_bound, compression="gzip")
            f.create_dataset("states", data=states, compression="gzip")
            f.create_dataset("actions", data=actions, compression="gzip")
            f.create_dataset("frames", data=images, compression="gzip")
            f.create_dataset("qpos", data=qposes,compression="gzip")

if __name__ == "__main__":
    """
    Generates masks for each robot, and stores it in the target directory.
    """
    # generate_sawyer_data()
    generate_all_sawyer_data()
    # generate_baxter_data()
    # generate_widowx_data()
