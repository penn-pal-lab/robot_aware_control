import os

from scipy.spatial.transform.rotation import Rotation
from src.env.robotics.rotations import euler2quat, mat2quat, quat2euler
from src.env.robotics.robot_env import RobotEnv
import numpy as np
import time
import imageio


class FetchMaskEnv(RobotEnv):
    def __init__(self):
        model_path = os.path.join("fetch", "robot_mask.xml")
        initial_qpos = None
        n_actions = 1
        n_substeps = 1
        seed = None
        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
        self._img_width = 320
        self._img_height = 240
        self._camera_name = "main_cam"
        self._joints = [
            "robot0:shoulder_pan_joint",
            "robot0:shoulder_lift_joint",
            "robot0:upperarm_roll_joint",
            "robot0:elbow_flex_joint",
            "robot0:forearm_roll_joint",
            "robot0:wrist_flex_joint",
            "robot0:wrist_roll_joint",
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

    def compare_traj(self, traj_name):
        # load the joint configuration and eef position
        path = f"robonet_images/berkeley_sawyer/qposes_{traj_name}.npy"
        data = np.load(path)
        gripper_path = f"robonet_images/berkeley_sawyer/states_{traj_name}.npy"
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
        while True:
            for i, qpos in enumerate(data):
                # grip_state = gripper_data[i]
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

                self.sim.forward()
                self.render("human")
                # img = self.render("rgb_array")
                # mask = self.get_robot_mask()
                # real_img = imageio.imread(f"robonet_images/berkeley_sawyer/{traj_name}_{i}.png")
                # mask_img = real_img.copy()
                # mask_img[mask] = (0, 255, 255)
                # # imageio.imwrite("mask_img.png", mask_img)
                # # import ipdb; ipdb.set_trace()
                # comparison = np.concatenate([img, real_img, mask_img], axis=1)
                # gif.append(comparison)
            # imageio.mimwrite(f"{traj_name}_mask.gif", gif)

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

    """
    calibrated camera intrinsic:
    [[320.75   0.   160.  ]
    [  0.   320.75 120.  ]
    [  0.     0.     1.  ]]
    calibrated camera extrinsic:
    [[-0.00938149  0.99966125  0.02427719 -0.34766725]
    [ 0.65135595  0.02453023 -0.7583757  -0.53490458]
    [-0.75871432  0.0086984  -0.65136543  1.08981838]
    [ 0.          0.          0.          1.        ]]
    calibrated projection matrix:
    [[-1.24403405e+02  3.22033088e+02 -9.64315591e+01  6.28566718e+01]
    [ 1.17876702e+02  8.91188007e+00 -3.21412856e+02 -4.07924397e+01]
    [-7.58714318e-01  8.69839730e-03 -6.51365428e-01  1.08981838e+00]]
    calibrated camera to world transformation:
    [[-0.00938149  0.65135595 -0.75871432  1.17201246]
    [ 0.99966125  0.02453023  0.0086984   0.35119113]
    [ 0.02427719 -0.7583757  -0.65136543  0.31265177]
    [ 0.          0.          0.          1.        ]]
    camera 3d position:
    [1.17201246 0.35119113 0.31265177]
    camera orientation (quarternion):
    [ 0.63589577  0.64909113 -0.28874116 -0.30157226]
    """
    traj_id = 5214
    traj_name = f"berkeley_sawyer_traj{traj_id}"
    camera_extrinsics = np.array(
        [
            [-0.00715332, 0.65439626, -0.75611796, 1.13910297],
            [0.9996319, 0.02446862, 0.01171972, 0.34967541],
            [0.0261705, -0.7557558, -0.65433041, 0.28774818],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    rot_matrix = camera_extrinsics[:3, :3]
    cam_pos = camera_extrinsics[:3, 3]
    rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
    cam_rot = Rotation.from_matrix(rot_matrix)

    env = FetchMaskEnv()
    cam_id = env.sim.model.camera_name2id("main_cam")
    env.sim.model.cam_pos[cam_id] = cam_pos
    cam_quat = cam_rot.as_quat()
    cam_quat = [
        cam_quat[3],
        cam_quat[0],
        cam_quat[1],
        cam_quat[2],
    ]
    env.sim.model.cam_quat[cam_id] = cam_quat

    site_id = env.sim.model.site_name2id("main_cam_site")
    env.sim.model.site_pos[site_id] = cam_pos
    env.sim.model.site_quat[site_id] = cam_quat
    env.sim.forward()

    env.compare_traj(traj_name)

    """
    Scene Visualization
    press tab to cycle through cameras. There is the default camera, and the main_cam which we set the pose of in baxter/robot.xml
    """
    # get openCV camera geom pose
    # rot_matrix = camera_extrinsics[:3, :3]
    # cam_pos = camera_extrinsics[:3, 3]
    # print("openCV cam pos", cam_pos)
    # cam_rot = Rotation.from_matrix(rot_matrix)
    # q = cam_rot.as_quat()
    # cam_quat = [q[3], q[0], q[1], q[2]]  # wxyz order
    # print("openCV cam quat", cam_quat)

    # get mujoco camera geom pose
    # print("mj cam pos", env.sim.model.cam_pos[cam_id])
    # print("mj cam quat", env.sim.model.cam_quat[cam_id])
    # while True:
    # env.render("human")
