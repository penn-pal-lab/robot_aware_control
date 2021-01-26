import os
from src.env.robotics.masks.base_mask_env import MaskEnv
import ipdb

from scipy.spatial.transform.rotation import Rotation
import numpy as np
import time
import imageio


class BaxterMaskEnv(MaskEnv):
    def __init__(self):
        model_path = os.path.join("baxter", "robot.xml")
        initial_qpos = None
        n_actions = 1
        n_substeps = 1
        seed = None
        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
        self._img_width = 160
        self._img_height = 120
        self._camera_name = "main_cam"
        self._joints = ["s0", "s1", "e0", "e1", "w0", "w1", "w2"]
        self._r_gripper_joints = [
            "r_gripper_l_finger_joint",
            "r_gripper_r_finger_joint",
        ]
        self._l_gripper_joints = [
            "l_gripper_l_finger_joint",
            "l_gripper_r_finger_joint",
        ]
        self.arm = "right"

    def generate_masks(self,  qpos_data):
        joint_references = [
            self.sim.model.get_joint_qpos_addr(f"{self.arm}_{x}") for x in self._joints
        ]
        masks = []
        for qpos in qpos_data:
            self.sim.data.qpos[joint_references] = qpos
            self.sim.forward()
            mask = self.get_robot_mask()
            masks.append(mask)
        masks = np.asarray(masks, dtype=np.bool)
        return masks

    def compare_traj(self, traj_name, arm, qpos_data, gripper_data, real_imgs):
        # load the joint configuration and eef position
        # path = f"robonet_images/penn_baxter_{arm}/qposes_{traj_name}.npy"
        # data = np.load(path)
        # gripper_path = f"robonet_images/penn_baxter_{arm}/states_{traj_name}.npy"
        # gripper_data = np.load(gripper_path)

        # baxter left 14
        # ws_min = np.array([0.45, 0.15, -0.15, 15.0, 0.0])
        # ws_max = np.array([0.75, 0.59, -0.05, 320, 100])

        # baxter right 1444
        # ws_min = np.array([0.4, -0.63, -0.15, 15.0, 0.0])
        # ws_max = np.array([0.75, -0.2, -0.05, 320, 100])
        # grip_force_min = ws_min[-1]
        # grip_force_max = ws_max[-1]
        joint_references = [
            self.sim.model.get_joint_qpos_addr(f"{arm}_{x}") for x in self._joints
        ]
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
            # grip_force /= (grip_force_max - grip_force_min)
            # # print(grip_force, ws_min[-1], ws_max[-1])
            # ipdb.set_trace()
            # assert -1 <= grip_force <= 1, grip_force
            # # https://github.com/ARISE-Initiative/robosuite/blob/v0.3/robosuite/models/grippers/two_finger_gripper.py
            # grip_motor_forces = np.array([grip_force, -grip_force])
            # rescale normalized action to control ranges
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
            eef_pos = grip_state[:3].copy()
            # eef_pos *= ws_max[:3] - ws_min[:3]
            # eef_pos += ws_min[:3]

            eef_site = self.sim.model.body_name2id("eef_body")
            self.sim.model.body_pos[eef_site] = eef_pos
            # print(eef_pos)

            self.sim.forward()
            # self.render("human")
            # img = self.render("rgb_array")
            mask = self.get_robot_mask()
            real_img = real_imgs[i]
            mask_img = real_img.copy()
            mask_img[mask] = (0, 255, 255)
            # imageio.imwrite("mask_img.png", mask_img)
            # import ipdb; ipdb.set_trace()
            comparison = mask_img
            # comparison = np.concatenate([img, real_img, mask_img], axis=1)
            gif.append(comparison)
        imageio.mimwrite(f"{traj_name}_mask.gif", gif)

    def _sample_goal(self):
        pass

    def _get_obs(self):
        return {"observation": np.array([0])}

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
                if any([a, b, c]):
                    mask[ids == i] = True
        return mask

if __name__ == "__main__":
    import pandas as pd
    from robonet.robonet.datasets.util.hdf5_loader import default_loader_hparams, load_data_customized
    from robonet.robonet.datasets.util.metadata_helper import load_metadata
    import tensorflow as tf
    from tqdm import tqdm

    robonet_root = "/media/ed/hdd/Datasets/Robonet/hdf5/"
    metadata_path = "/media/ed/hdd/Datasets/Robonet/hdf5/meta_data.pkl"
    arm = "right"
    num_test = 100
    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [120, 160]
    hparams.cams_to_load = [0]

    df = pd.read_pickle(metadata_path, compression="gzip")
    baxter_df = df.loc["baxter" == df["robot"]]
    left = ["left" in x for x in baxter_df.index]
    right = [not x for x in left]
    baxter_subset = baxter_df[left if arm == "left" else right]

    if arm == "right":
        camera_extrinsics = np.array(
            [
                [0.59474902, -0.48560866, 0.64066983, 0.00593267],
                [-0.80250365, -0.40577623, 0.4374169, -0.84046503],
                [0.04755516, -0.77429315, -0.63103774, 0.45875102],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    else:
        camera_extrinsics = np.array(
            [
                [0.05010049, 0.5098481, -0.85880432, 1.70268951],
                [0.99850135, -0.00660876, 0.05432662, 0.26953027],
                [0.02202269, -0.86023906, -0.50941512, 0.48536055],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    rot_matrix = camera_extrinsics[:3, :3]
    cam_pos = camera_extrinsics[:3, 3]
    rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
    cam_rot = Rotation.from_matrix(rot_matrix) * rel_rot
    env = BaxterMaskEnv()
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

    rand_baxter = baxter_subset.sample(num_test)
    meta_data = load_metadata(robonet_root)
    # load qpos, gripper states, workspace bounds
    for traj_name in tqdm(rand_baxter.index, "generating gifs"):
        f_metadata = df.loc[traj_name]
        f_name = os.path.join(robonet_root, traj_name)
        data = load_data_customized(f_name, f_metadata, hparams)
        imgs, states, qposes, ws_min, ws_max, vp = data
        imgs = imgs[:, 0]
        env.compare_traj(traj_name, arm, qposes, states, imgs)
