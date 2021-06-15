import os
import h5py
from scipy.spatial.transform.rotation import Rotation
from src.env.robotics.masks.base_mask_env import MaskEnv
from tqdm import tqdm
import numpy as np
import time
import imageio


class FrankaMaskEnv(MaskEnv):
    def __init__(self):
        model_path = os.path.join("franka", "robot.xml")
        initial_qpos = None
        n_actions = 1
        n_substeps = 1
        seed = None
        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
        self._img_width = 64
        self._img_height = 48
        self._camera_name = "main_cam"
        self._joints = [f"joint{i}" for i in range(1,8)]
        # self._r_gripper_joints = [
        #     "r_gripper_l_finger_joint",
        #     "r_gripper_r_finger_joint",
        # ]
        # self._l_gripper_joints = [
        #     "l_gripper_l_finger_joint",
        #     "l_gripper_r_finger_joint",
        # ]

    def compare_traj(self, traj_name, qpos_data, gripper_data, real_imgs):
        # load the joint configuration and eef position
        # path = f"robonet_images/berkeley_sawyer/qposes_{traj_name}.npy"
        # data = np.load(path)
        # gripper_path = f"robonet_images/berkeley_sawyer/states_{traj_name}.npy"
        # gripper_data = np.load(gripper_path)
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

    def get_robot_mask(self, width=None, height=None):
        """
        Return binary img mask where 1 = robot and 0 = world pixel.
        robot_mask_with_obj means the robot mask is computed with object occlusions.
        """
        if width is None or height is None:
            mask_dim = [self._img_height, self._img_width]
        else:
            mask_dim = [height, width]
        # returns a binary mask where robot pixels are True
        seg = self.render("rgb_array", segmentation=True, width=width, height=height)  # flip the camera
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == self.mj_const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
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
                d = "finger" in name
                if any([a, b, c, d]):
                    mask[ids == i] = True
        return mask


if __name__ == "__main__":
    VISUALIZE = False
    DATA_ROOT = "/home/pallab/locobot_ws/src/eef_control/data/franka_views/c0"
    camera_extrinsics = np.array(
        [[-0.00589602,  0.76599739, -0.64281664,  1.11131074],
        [ 0.9983059,  -0.03270131, -0.04812437,  0.07869842],
        [-0.05788409, -0.64201138, -0.76450691,  0.59455265],
        [ 0.,          0.,          0.,          1.        ]]
    )
    # offset = [0.01, -0.02, 0.02]
    offset = [0.0, -0.01, 0.02]
    rot_matrix = camera_extrinsics[:3, :3]
    cam_pos = camera_extrinsics[:3, 3]
    rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
    cam_rot = Rotation.from_matrix(rot_matrix)

    env = FrankaMaskEnv()
    env.set_opencv_camera_pose("main_cam", camera_extrinsics, offset)

    # load the franka qpos data and generate mask
    files = []
    for d in os.scandir(DATA_ROOT):
        if d.is_file() and d.path.endswith("hdf5"):
            files.append(d.path)

    for fp in tqdm(files):
        with h5py.File(fp, "r") as hf:
            qpos = hf["qpos"][:]
            states = hf["states"][:]
            imgs = hf["observations"][:]

        if VISUALIZE:
            masks = env.generate_masks(qpos, 640, 480)
            gif = []
            for i in range(len(imgs)):
                img = imgs[i]
                mask = masks[i]
                img[mask] = (0, 255, 0)
                gif.append(img)
            imageio.mimwrite(f"{fp}.gif", gif)

            # while True:
            #     env.render("human")
        else:
            masks = env.generate_masks(qpos, 64, 48)
            with h5py.File(fp, "a") as hf:
                if "masks" in hf.keys():
                    hf["masks"][:] = masks
                else:
                    hf.create_dataset("masks", data=masks)
