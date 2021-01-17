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
        self._joints = ["s0", "s1", "e0", "e1", "w0", "w1", "w2"]
        self._r_gripper_joints = [
            "r_gripper_l_finger_joint",
            "r_gripper_r_finger_joint",
        ]
        self._l_gripper_joints = [
            "l_gripper_l_finger_joint",
            "l_gripper_r_finger_joint",
        ]

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

    def compare_traj(self, traj_name, arm):
        # load the joint configuration and eef position
        path = f"robonet_images/qposes_{traj_name}.npy"
        data = np.load(path)
        gripper_path = f"robonet_images/states_{traj_name}.npy"
        gripper_data = np.load(gripper_path)
        grip_force_min = 0
        grip_force_max = 100
        joint_references = [
            self.sim.model.get_joint_qpos_addr(f"{arm}_{x}") for x in self._joints
        ]
        if arm == "right":
            gripper_references = [
                self.sim.model.get_joint_qpos_addr(x) for x in self._r_gripper_joints
            ]
        else:
            gripper_references = [
                self.sim.model.get_joint_qpos_addr(x) for x in self._l_gripper_joints
            ]
        # run qpos trajectory
        gif = []
        for i, qpos in enumerate(data):
            grip_state = gripper_data[i]
            grip_force = grip_state[-1]
            grip_force -= grip_force_min
            grip_force /= grip_force_max - grip_force_min
            assert -1 <= grip_force <= 1
            # https://github.com/ARISE-Initiative/robosuite/blob/v0.3/robosuite/models/grippers/two_finger_gripper.py
            grip_motor_forces = np.array([grip_force, -grip_force])
            # rescale normalized action to control ranges
            if arm == "right":
                ctrl_range = self.sim.model.actuator_ctrlrange
                bias = 0.5 * (ctrl_range[-4:-2, 1] + ctrl_range[-4:-2, 0])
                weight = 0.5 * (ctrl_range[-4:-2, 1] - ctrl_range[-4:-2, 0])
            else:
                ctrl_range = self.sim.model.actuator_ctrlrange
                bias = 0.5 * (ctrl_range[-2:, 1] + ctrl_range[-2:, 0])
                weight = 0.5 * (ctrl_range[-2:, 1] - ctrl_range[-2:, 0])
            grip_motor_forces = bias + weight * grip_motor_forces
            init_grip = [-0.015, 0.015]
            grip_motor_forces += init_grip

            self.sim.data.qpos[joint_references] = qpos
            self.sim.data.qpos[gripper_references] = grip_motor_forces

            self.sim.forward()
            img = self.render("rgb_array")
            mask = self.get_robot_mask()
            real_img = imageio.imread(f"robonet_images/{traj_name}_{i}.png")
            mask_img = real_img.copy()
            mask_img[mask] = (0, 255, 255)
            # imageio.imwrite("mask_img.png", mask_img)
            # import ipdb; ipdb.set_trace()
            comparison = np.concatenate([img, real_img, mask_img], axis=1)
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
                d = "finger" in name
                if any([a, b, c, d]):
                    mask[ids == i] = True
        return mask


if __name__ == "__main__":

    """
    Input: Camera Extrinsics Matrix
    Given the openCV camera extrinsics, we would like to replicate the camera pose in mujoco.
    To do so, we have to flip the camera's x axis orientation

    I have calculated the relative orientation shift, so we can automatically do this conversion in the future. The relative orientation quat is [0 1 0 0]
    """

    """Right Arm Matrix
        camera_extrinsics = np.array(
        [
            [0.59474902, -0.48560866, 0.64066983, 0.00593267],
            [-0.80250365, -0.40577623, 0.4374169, -0.84046503],
            [0.04755516, -0.77429315, -0.63103774, 0.45875102],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    """

    """Left Arm Matrix
    camera_extrinsics = np.array(
        [
            [0.05010049, 0.5098481, -0.85880432, 1.70268951],
            [0.99850135, -0.00660876, 0.05432662, 0.26953027],
            [0.02202269, -0.86023906, -0.50941512, 0.48536055],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    """
    arm = "left"
    traj_id = 14
    traj_name = f"penn_baxter_{arm}_traj{traj_id}"
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

    # rot_matrix[:, 0] *= -1 # flip the x axis
    # cam_rot = Rotation.from_matrix(rot_matrix)

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
    env.compare_traj(traj_name, arm)

    """
    Scene Visualization
    press tab to cycle through cameras. There is the default camera, and the main_cam which we set the pose of in baxter/robot.xml
    """
    # # get openCV camera geom pose
    # rot_matrix = camera_extrinsics[:3, :3]
    # cam_pos = camera_extrinsics[:3, 3]
    # print("openCV cam pos", cam_pos)
    # cam_rot = Rotation.from_matrix(rot_matrix)
    # q = cam_rot.as_quat()
    # cam_quat = [q[3], q[0], q[1], q[2]]  # wxyz order
    # print("openCV cam quat", cam_quat)

    # # get mujoco camera geom pose
    # print("mj cam pos", env.sim.model.cam_pos[cam_id])
    # print("mj cam quat", env.sim.model.cam_quat[cam_id])
    # while True:
    #     env.render("human")
