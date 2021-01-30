import os
from src.env.robotics.masks.base_mask_env import MaskEnv

import imageio
import numpy as np



class SawyerMaskEnv(MaskEnv):
    def __init__(self):
        model_path = os.path.join("sawyer", "robot.xml")
        initial_qpos = None
        n_actions = 1
        n_substeps = 1
        seed = None
        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
        self._img_width = 85
        self._img_height = 64
        self._camera_name = "main_cam"
        self._joints = [f"right_j{i}" for i in range(7)]

    def compare_traj(self, traj_name, qpos_data, gripper_data, real_imgs):
        # load the joint configuration and eef position
        # path = f"robonet_images/berkeley_sawyer/qposes_{traj_name}.npy"
        # data = np.load(path)
        # gripper_path = f"robonet_images/berkeley_sawyer/states_{traj_name}.npy"
        # gripper_data = np.load(gripper_path)
        # visualize the eef site
        # ws_min = np.array([0.45, -0.18, 0.176, 1.57079633, 1.0])
        # ws_max = np.array([0.79, 0.22, 0.292, 4.62512252, 1.0])

        # grip_force_min = 0
        # grip_force_max = 100
        joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]
        # if arm == "right":
        #     gripper_references = [
        #         self.sim.model.get_joint_qpos_addr(x) for x in self._r_gripper_joints
        #     ]
        # else:
        #     gripper_references = [
        #         self.sim.model.get_joint_qpos_addr(x) for x in self._l_gripper_joints
        #     ]
        # run qpos trajectory
        gif = []
        for i, qpos in enumerate(qpos_data):
            grip_state = gripper_data[i]
            # grip_force = grip_state[-1]
            # grip_force -= grip_force_min
            # grip_force /= grip_force_max - grip_force_min
            # assert -1 <= grip_force <= 1
            # # https://github.com/ARISE-Initiative/robosuite/blob/v0.3/robosuite/models/grippers/two_finger_gripper.py
            # grip_motor_forces = np.array([grip_force, -grip_force])
            # # rescale normalized action to control ranges
            # if arm == "right":
            #     ctrl_range = self.sim.model.actuator_ctrlrange
            #     bias = 0.5 * (ctrl_range[-4:-2, 1] + ctrl_range[-4:-2, 0])
            #     weight = 0.5 * (ctrl_range[-4:-2, 1] - ctrl_range[-4:-2, 0])
            # else:
            #     ctrl_range = self.sim.model.actuator_ctrlrange
            #     bias = 0.5 * (ctrl_range[-2:, 1] + ctrl_range[-2:, 0])
            #     weight = 0.5 * (ctrl_range[-2:, 1] - ctrl_range[-2:, 0])
            # grip_motor_forces = bias + weight * grip_motor_forces
            # init_grip = [-0.015, 0.015]
            # grip_motor_forces += init_grip
            self.sim.data.qpos[joint_references] = qpos
            # self.sim.data.qpos[gripper_references] = grip_motor_forces
            eef_pos = grip_state[:3]
            # eef_pos *= (ws_max[:3] - ws_min[:3])
            # eef_pos += ws_min[:3]

            eef_site = self.sim.model.body_name2id("eef_body")
            self.sim.model.body_pos[eef_site] = eef_pos
            self.sim.forward()
            # self.render("human")
            # img = self.render("rgb_array")
            mask = self.get_robot_mask()
            real_img = real_imgs[i]
            mask_img = real_img.copy()
            mask_img[mask] = (0, 255, 255)
            # mask_img[mask] = img[mask]
            # imageio.imwrite("mask_img.png", mask_img)
            # import ipdb; ipdb.set_trace()
            comparison = mask_img
            # comparison = np.concatenate([img, real_img, mask_img], axis=1)
            gif.append(comparison)
        imageio.mimwrite(f"{traj_name}_mask.gif", gif)


    def _is_robot_geom(self, name):
        geom = False
        ignore_parts = {"base_vis", "base_col", "head_vis"}
        if name in ignore_parts:
            return False
        a = "vis" in name
        b = "col" in name
        c = "gripper" in name
        d = "wsg" in name
        if any([a, b, c, d]):
            geom = True
        return geom



if __name__ == "__main__":
    import pandas as pd
    import tensorflow as tf
    from robonet.robonet.datasets.util.hdf5_loader import (
        default_loader_hparams, load_data_customized)
    from robonet.robonet.datasets.util.metadata_helper import load_metadata
    from tqdm import tqdm

    robonet_root = "/media/ed/hdd/Datasets/Robonet/hdf5/"
    metadata_path = "/media/ed/hdd/Datasets/Robonet/hdf5/meta_data.pkl"
    arm = "right"
    num_test = 10
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
    rand_sawyer = sawyer_subset.sample(num_test)
    meta_data = load_metadata(robonet_root)
    # load qpos, gripper states, workspace bounds
    for traj_name in tqdm(rand_sawyer.index, "generating gifs"):
        f_metadata = df.loc[traj_name]
        f_name = os.path.join(robonet_root, traj_name)
        data = load_data_customized(f_name, f_metadata, hparams)
        imgs, states, qposes, ws_min, ws_max, vp = data
        imgs = imgs[:, 0]
        env.compare_traj(traj_name, qposes, states, imgs)
