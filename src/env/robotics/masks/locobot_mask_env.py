import os
from src.env.robotics.masks.base_mask_env import MaskEnv

from scipy.spatial.transform.rotation import Rotation
from src.env.robotics.rotations import euler2quat, mat2quat, quat2euler
from src.env.robotics.robot_env import RobotEnv
import numpy as np
import time
import imageio


class LocobotMaskEnv(MaskEnv):
    def __init__(self):
        # TODO: change the model path
        model_path = os.path.join("locobot", "locobot.xml")
        initial_qpos = None
        n_actions = 1
        n_substeps = 1
        seed = None
        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
        self._img_width = 85
        self._img_height = 64
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

    def get_robot_mask(self, width=None, height=None):
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
        if width is None or height is None:
            mask_dim = [self._img_height, self._img_width]
        else:
            mask_dim = [height, width]
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

    env = LocobotMaskEnv()
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

    while True:
        env.render("human")
