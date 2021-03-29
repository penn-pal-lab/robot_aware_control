import os
from src.env.robotics.controllers.sawyer_ik_controller import SawyerIKController
from src.env.robotics.masks.base_mask_env import MaskEnv

import imageio
import numpy as np
import ipdb

from gym.envs.robotics.rotations import (
    euler2quat,
    mat2euler,
    quat2mat,
    quat2euler,
    mat2quat,
)

import src.env.robotics.controllers.transform_utils as T


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
        bullet_data_path = os.path.join(
            os.path.dirname(__file__), "../assets/bullet_data"
        )
        self._joint_references = [
            self.sim.model.get_joint_qpos_addr(x) for x in self._joints
        ]
        self._controller = SawyerIKController(
            bullet_data_path=bullet_data_path, robot_jpos_getter=self.robot_jpos_getter
        )

    def robot_jpos_getter(self):
        return self.sim.data.qpos[self._joint_references].copy()

    def _bounded_d_pos(self, d_pos, pos):
        """
        Clips d_pos to the gripper limits
        """
        min_action = self._min_gripper_pos - pos
        max_action = self._max_gripper_pos - pos
        return np.clip(d_pos, min_action, max_action)

    def compare_traj(self, traj_name, qpos_data, gripper_data, real_imgs, actions, ws_min, ws_max):
        self._min_gripper_pos = ws_min[:3]
        self._max_gripper_pos = ws_max[:3]
        # run qpos trajectory
        gif = []
        true_gif = []

        # initialize controller to first state
        self.sim.data.qpos[self._joint_references] = qpos_data[0].copy()
        self.sim.forward()
        self._controller.sync_state()
        ik_masks = [self.get_robot_mask()]
        # start using actions to predict future state, and run IK to get mask
        for i, ac in enumerate(actions):
            ac = gripper_data[i+1] - gripper_data[i]
            gripper_pos = self.sim.data.get_body_xpos("right_hand")
            desired_pos =  self._bounded_d_pos(ac[:3], gripper_pos)

            print("Desired pos", gripper_pos + desired_pos)

            # self._initial_right_hand_quat = T.euler_to_quat(action[3:6] * self._rotate_speed, self._initial_right_hand_quat)
            # d_quat = T.quat_multiply(T.quat_inverse(self._right_hand_quat), self._initial_right_hand_quat)

            euler_ac = [ac[3], 0, 0]
            desired_quat = T.euler_to_quat(
                euler_ac, self._right_hand_quat
            )
            d_quat = T.quat_multiply(T.quat_inverse(self._right_hand_quat), desired_quat)
            desired_ori = T.quat2mat(d_quat)
            joint_angles = self._controller.joint_positions_for_eef_command(desired_pos, desired_ori)

            # render the IK qpos
            self.sim.data.qpos[self._joint_references] = joint_angles
            self.sim.forward()
            print("IK's final pos", self.sim.data.get_body_xpos("right_hand"))
            # ik_img = self.render("rgb_array")
            ik_mask = self.get_robot_mask()
            ik_masks.append(ik_mask)


        # generate images with the ik mask overlay on the true qpos data
        for i, qpos in enumerate(qpos_data):
            grip_state = gripper_data[i]
            self.sim.data.qpos[self._joint_references] = qpos
            eef_pos = grip_state[:3]
            eef_site = self.sim.model.body_name2id("eef_body")
            self.sim.model.body_pos[eef_site] = eef_pos
            self.sim.forward()
            # self.render("human")
            # img = self.render("rgb_array")
            mask = ik_masks[i]
            ik_img = real_imgs[i].copy()
            ik_img[mask] = (0, 255, 255)

            real_img = real_imgs[i].copy()
            mask = self.get_robot_mask()
            real_img[mask] = (0, 255, 255)
            # mask_img[mask] = (0, 255, 255)
            comparison = np.concatenate([ik_img, real_img], 1)
            true_gif.append(comparison)
        imageio.mimwrite(f"{traj_name}_ik_mask.gif", true_gif)

        return

        # run the true qpos and generate a gif for reference
        for i, qpos in enumerate(qpos_data):
            grip_state = gripper_data[i]
            self.sim.data.qpos[self._joint_references] = qpos
            eef_pos = grip_state[:3]
            eef_site = self.sim.model.body_name2id("eef_body")
            self.sim.model.body_pos[eef_site] = eef_pos
            self.sim.forward()
            # self.render("human")
            # img = self.render("rgb_array")
            mask = self.get_robot_mask()
            real_img = real_imgs[i]
            mask_img = real_img.copy()
            mask_img[mask] = (0, 255, 255)
            comparison = mask_img
            true_gif.append(comparison)
        imageio.mimwrite(f"{traj_name}_mask.gif", gif)

    def _is_robot_geom(self, name):
        geom = False
        ignore_parts = {"base_vis", "base_col", "head_vis"}
        if name in ignore_parts:
            return False
        # return True
        a = "vis" in name
        b = "col" in name
        c = "gripper" in name
        d = "wsg" in name
        e = "right" in name
        if any([a, b, c, d, e]):
            geom = True
        return geom

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_quat(self):
        """
        Returns eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    def get_gripper_pos(self, qpos):
        self.sim.data.qpos[self._joint_references] = qpos
        self.sim.forward()
        return self.sim.data.get_body_xpos("right_hand").copy()

if __name__ == "__main__":
    import pandas as pd
    import tensorflow as tf
    from robonet.robonet.datasets.util.hdf5_loader import (
        default_loader_hparams,
        load_trajectory,
    )
    from robonet.robonet.datasets.util.metadata_helper import load_metadata
    from tqdm import tqdm
    from src.utils.camera_calibration import camera_to_world_dict

    robonet_root = "/media/ed/hdd/Datasets/Robonet/hdf5/"
    metadata_path = "/media/ed/hdd/Datasets/Robonet/hdf5/meta_data.pkl"
    arm = "right"
    num_test = 6
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [120, 160]
    hparams.cams_to_load = [0]

    df = pd.read_pickle(metadata_path, compression="gzip")
    sawyer_df = df.loc["sawyer" == df["robot"]]
    sawyer_subset = sawyer_df["sudri0" == sawyer_df["camera_configuration"]]

    camera_extrinsics = camera_to_world_dict[
        f"sawyer_sudri0_c{hparams.cams_to_load[0]}"
    ]
    env = SawyerMaskEnv()
    env.set_opencv_camera_pose("main_cam", camera_extrinsics)

    rand_sawyer = sawyer_subset.sample(num_test)
    # rand_sawyer = sawyer_subset
    meta_data = load_metadata(robonet_root)
    # load qpos, gripper states, workspace bounds
    for traj_name in tqdm(rand_sawyer.index, "generating gifs"):
        f_metadata = df.loc[traj_name]
        f_name = os.path.join(robonet_root, traj_name)
        data = load_trajectory(f_name, f_metadata, hparams)
        imgs, states, qposes, actions, ws_min, ws_max, vp = data
        imgs = imgs[:, 0]
        env.compare_traj(traj_name, qposes, states, imgs, actions, ws_min, ws_max)
