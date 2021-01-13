import os

from scipy.spatial.transform.rotation import Rotation
from src.env.fetch.rotations import euler2quat, mat2quat, quat2euler
from src.env.fetch.robot_env import RobotEnv
import numpy as np
import time
import imageio


class BaxterEnv(RobotEnv):
    def __init__(self):
        model_path = os.path.join("baxter", "robot.xml")
        initial_qpos = None
        n_actions = 1
        n_substeps = 1
        seed = None
        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
        self._img_width = 320
        self._img_height = 240
        self._camera_name = "main_cam"

    def render(self, mode, segmentation=False):
        if mode == "rgb_array":
            out = super().render(
                mode,
                width=self._img_width,
                height=self._img_height,
                camera_name=self._camera_name,
                segmentation=segmentation,
            )
            return out[::-1, ::-1]
        elif mode == "human":
            super().render(mode)

    def compare_traj(self, traj_name):
        # load the joint configuration and eef position
        path = f"robonet_images/qposes_{traj_name}.npy"
        data = np.load(path)
        # set camera position
        # run qpos trajectory
        gif = []
        while True:
            for i, qpos in enumerate(data):
                self.sim.data.qpos[7:] = qpos
                self.sim.forward()
                img = self.render("rgb_array")
                mask = self.get_robot_mask()
                real_img = imageio.imread(f"robonet_images/{traj_name}_{i}.png")
                mask_img = real_img.copy()
                mask_img[mask] = img[mask]
                # imageio.imwrite("mask_img.png", mask_img)
                # import ipdb; ipdb.set_trace()
                comparison = np.concatenate([img, real_img, mask_img], axis=1)
                gif.append(comparison)
            imageio.mimwrite(f"{traj_name}_mask.gif", gif)
            break

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
                if any([a, b]):
                    mask[ids == i] = True
        return mask


if __name__ == "__main__":

    """
    Input: Camera Extrinsics Matrix
    Given the openCV camera extrinsics, we would like to replicate the camera pose in mujoco.
    To do so, we have to flip the camera's x axis orientation, and add 0.062 to the z position.

    I have calculated the relative orientation shift, so we can automatically do this conversion in the future. The relative orientation quat is [0 1 0 0]
    """


    camera_extrinsics = np.array(
        [
            [0.06013387, 0.53578203, -0.84221229, 1.59074811],
            [0.99793479, -0.01317843, 0.06286884, 0.2689595],
            [0.02258496, -0.84425348, -0.535468, 0.4451598],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rot_matrix = camera_extrinsics[:3, :3]
    cam_pos = camera_extrinsics[:3,3]
    rel_rot = Rotation.from_quat([0,1,0,0]) # calculated
    cam_rot = Rotation.from_matrix(rot_matrix) * rel_rot

    # cam_euler = quat2euler([-0.35759175, 0.63418852, 0.60459815, -0.32310082])
    # # cam_euler[1] += pi
    # print("camera euler: ", cam_euler)

    env = BaxterEnv()
    # # calculate relative quaternion between old mujoco quat and main_cam quaternion
    # # wxyz quat
    # old_quat = [-0.35759175, 0.63418852, 0.60459815, -0.32310082]
    # old_rot = Rotation.from_quat([0.63418852, 0.60459815, -0.32310082, -0.35759175])

    # env.sim.forward()
    # # wxyz quat
    # new_quat = mat2quat(env.sim.data.cam_xmat[0].reshape(3,3))
    # new_rot = Rotation.from_quat([new_quat[1], new_quat[2], new_quat[3], new_quat[0]])
    # # new_rot = old_rot * rel_rot
    # # old_rot^-1 * new_rot = rel_rot
    # rel_rot = old_rot.inv() * new_rot
    # rel_quat = rel_rot.as_quat()
    # # rel_quat is [0 1 0 0] wxyz
    # mujoco_rel_quat = [rel_quat[3], rel_quat[0], rel_quat[1], rel_quat[2]]

    # cv_rot = Rotation.from_matrix(rot_matrix)
    # muj_rot = cv_rot * rel_rot
    # muj_quat = muj_rot.as_quat()
    # muj_quat = [muj_quat[3], muj_quat[0], muj_quat[1], muj_quat[2]]


    # env.sim.forward()
    # curr_quat = mat2quat(env.sim.data.cam_xmat[0].reshape(3,3))
    # print(muj_quat)
    # print(curr_quat)
    cam_id = 0
    env.sim.model.cam_pos[cam_id] = cam_pos + [0, 0, 0.062]
    cam_quat = cam_rot.as_quat()
    env.sim.model.cam_quat[cam_id] = [cam_quat[3], cam_quat[0], cam_quat[1], cam_quat[2]]
    env.sim.forward()
    traj_name = "penn_baxter_left_traj78"
    env.compare_traj(traj_name)

    """
    Scene Visualization
    press tab to cycle through cameras. There is the default camera, and the main_cam which we set the pose of in baxter/robot.xml
    """
    # while True:
    #     env.render("human")
