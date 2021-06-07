from src.env.robotics.masks.locobot_mask_env import LocobotMaskEnv
from src.env.robotics.masks.franka_mask_env import FrankaMaskEnv
import h5py
import imageio
import numpy as np
import os
import sys

# CAMERA_CALIB = np.array(
#     [
#         [0.008716, 0.75080825, -0.66046272, 0.77440888],
#         [0.99985879, 0.00294645, 0.01654445, 0.02565873],
#         [0.01436773, -0.66051366, -0.75067655, 0.64211797],
#         [0.0, 0.0, 0.0, 1.0],
#     ]
# )

# for fp in os.scandir("/home/pallab/locobot_ws/src/eef_control/data/"):
#     if "teaser_start_locobot_" in fp.name:
#         print(fp.path)
#         LOCOBOT_FILE = fp.path
#         FILE_NAME = fp.name

#         with h5py.File(LOCOBOT_FILE, "r") as hf:
#             qpos = hf["qpos"][:][0]
#             img = hf["observations"][:][0]

#         # env = LocobotMaskEnv()
#         # offset = [-0.0,-0.040,0]
#         # env.set_opencv_camera_pose("main_cam", CAMERA_CALIB, offset)
#         # mask = env.generate_masks([qpos], 640, 480)[0]

#         # robot_img = img.copy()
#         # robot_img[~mask] = (0, 0, 0)
#         # world_img = img.copy()
#         # world_img[mask] = (0, 0, 0)
#         imageio.imwrite(f"{FILE_NAME}_orig_img.png", img)
#         # imageio.imwrite(f"{FILE_NAME}_robot_img.png", robot_img)
#         # imageio.imwrite(f"{FILE_NAME}_robot_mask.png", np.uint8(mask)* 255)
#         # imageio.imwrite(f"{FILE_NAME}_masked_world.png", world_img)

# FRANKA
CAMERA_CALIB = np.array(
        [
            [-0.00589602, 0.76599739, -0.64281664, 1.11131074],
            [0.9983059, -0.03270131, -0.04812437, 0.07869842 - 0.01],
            [-0.05788409, -0.64201138, -0.76450691, 0.59455265 + 0.02],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

env = FrankaMaskEnv()
offset = [0.0,0.01,0.0]
env.set_opencv_camera_pose("main_cam", CAMERA_CALIB, offset)
for fp in os.scandir("/home/pallab/locobot_ws/src/eef_control/data/"):
    if "teaser_vis" in fp.name:
        print(fp.path)
        LOCOBOT_FILE = fp.path
        FILE_NAME = fp.name

        with h5py.File(LOCOBOT_FILE, "r") as hf:
            qpos = hf["qpos"][:][0]
            img = hf["observations"][:][0]

        mask = env.generate_masks([qpos], 640, 480)[0]

        robot_img = img.copy()
        robot_img[~mask] = (0, 0, 0)
        world_img = img.copy()
        world_img[mask] = (0, 0, 0)
        imageio.imwrite(f"{FILE_NAME}_orig_img.png", img)
        imageio.imwrite(f"{FILE_NAME}_robot_img.png", robot_img)
        imageio.imwrite(f"{FILE_NAME}_robot_mask.png", np.uint8(mask)* 255)
        imageio.imwrite(f"{FILE_NAME}_masked_world.png", world_img)