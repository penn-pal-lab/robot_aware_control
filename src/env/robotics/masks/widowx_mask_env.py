import os
from src.env.robotics.masks.base_mask_env import MaskEnv

from scipy.spatial.transform.rotation import Rotation
from src.env.robotics.rotations import euler2quat, mat2quat, quat2euler
from src.env.robotics.robot_env import RobotEnv
import numpy as np
import time
import imageio


class WidowXMaskEnv(MaskEnv):
    def __init__(self):
        model_path = os.path.join("widowx", "robot.xml")
        initial_qpos = None
        n_actions = 1
        n_substeps = 1
        seed = None
        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
        self._img_width = 160
        self._img_height = 120
        self._camera_name = "main_cam"
        self._joints = [f"joint_{i}" for i in range(1, 6)]
        self._joints.append("gripper_revolute_joint")

    def compare_traj(self, traj_name, qpos_data, gripper_data, real_imgs):
        joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]
        # run qpos trajectory
        gif = []
        for i, qpos in enumerate(qpos_data):
            self.sim.data.qpos[joint_references] = qpos
            grip_state = gripper_data[i]
            eef_pos = grip_state[:3]
            eef_site = self.sim.model.body_name2id("eef_body")
            self.sim.model.body_pos[eef_site] = eef_pos
            self.sim.forward()
            # self.render("human")
            img = self.render("rgb_array")
            mask = self.get_robot_mask()
            real_img = real_imgs[i]
            mask_img = real_img.copy()
            mask_img[mask] = (0, 255, 255)
            comparison = mask_img
            # comparison = np.concatenate([img, real_img, mask_img], axis=1)
            gif.append(comparison)
        imageio.mimwrite(f"{traj_name}_mask.gif", gif)

    def get_robot_mask(self):
        """
        Return binary img mask where 1 = robot and 0 = world pixel.
        robot_mask_with_obj means the robot mask is computed with object occlusions.
        """
        # returns a binary mask where robot pixels are True
        seg = self.render("rgb_array", segmentation=True)  # flip the camera
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == self.mj_const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        mask_dim = [self._img_height, self._img_width]
        mask = np.zeros(mask_dim, dtype=np.bool)
        ignore_parts = {"base_link_vis", "base_link_col", "head_vis"}
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if name is not None:
                if name in ignore_parts:
                    continue
                a = "vis" in name
                b = "col" in name
                c = "gripper" in name
                d = "mesh" in name
                if any([a, b, c, d]):
                    mask[ids == i] = True
        return mask


if __name__ == "__main__":
    import pandas as pd
    from robonet.robonet.datasets.util.hdf5_loader import (
        default_loader_hparams,
        load_data_customized,
    )
    from robonet.robonet.datasets.util.metadata_helper import load_metadata
    import tensorflow as tf
    from tqdm import tqdm

    robonet_root = "/media/ed/hdd/Datasets/Robonet/hdf5/"
    metadata_path = "/media/ed/hdd/Datasets/Robonet/hdf5/meta_data.pkl"
    num_test = 10
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

    rot_matrix = camera_extrinsics[:3, :3]
    cam_pos = camera_extrinsics[:3, 3]
    rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
    cam_rot = Rotation.from_matrix(rot_matrix) * rel_rot
    env = WidowXMaskEnv()
    cam_id = 0
    offset = [0, 0, 0]
    env.sim.model.cam_pos[cam_id] = cam_pos + offset
    cam_quat = cam_rot.as_quat()
    env.sim.model.cam_quat[cam_id] = [
        cam_quat[3],
        cam_quat[0],
        cam_quat[1],
        cam_quat[2],
    ]
    env.sim.forward()

    rand_sawyer = widowx_subset.sample(num_test, random_state=5)
    meta_data = load_metadata(robonet_root)
    # load qpos, gripper states, workspace bounds
    for traj_name in tqdm(rand_sawyer.index, "generating gifs"):
        f_metadata = df.loc[traj_name]
        f_name = os.path.join(robonet_root, traj_name)
        data = load_data_customized(f_name, f_metadata, hparams)
        imgs, states, qposes, ws_min, ws_max, vp = data
        imgs = imgs[:, 0]
        env.compare_traj(traj_name, qposes, states, imgs)
