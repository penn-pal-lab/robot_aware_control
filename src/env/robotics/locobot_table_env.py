from collections import defaultdict
from src.env.robotics.masks.locobot_analytical_ik import AnalyticInverseKinematics
from src.env.robotics.utils import (
    mocap_set_action,
    reset_mocap2body_xpos,
    reset_mocap_welds,
)
from src.env.robotics.masks.base_mask_env import MaskEnv
from gym import spaces
import numpy as np
import os

DEFAULT_PITCH = 1.3
DEFAULT_ROLL = 0.0
CAMERA_CALIB = np.array(
    [
        [0.008716, 0.75080825, -0.66046272, 0.77440888],
        [0.99985879, 0.00294645, 0.01654445, 0.02565873],
        [0.01436773, -0.66051366, -0.75067655, 0.64211797],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class LocobotTableEnv(MaskEnv):
    def __init__(self, config):
        self._config = config
        modified =  config.modified
        model_path = f"locobot_table{'_modified' if modified else ''}.xml"
        model_path = os.path.join("locobot", model_path)

        initial_qpos = None
        n_actions = 3
        n_substeps = 20
        seed = None
        self._img_width = 64
        self._img_height = 48
        self._render_device = config.render_device
        self._joints = [f"joint_{i}" for i in range(1, 6)]

        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)

        self._camera_name = "main_cam"
        # self._joints.append("gripper_revolute_joint")
        self._joint_references = [
            self.sim.model.get_joint_qpos_addr(x) for x in self._joints
        ]
        self._joint_vel_references = [
            self.sim.model.get_joint_qvel_addr(x) for x in self._joints
        ]
        self._geoms = {
            # "robot-geom-0",
            # "robot-geom-1",
            # "robot-geom-2",
            # "robot-geom-3",
            # "robot-geom-4",
            # "robot-geom-5",
            # "robot-geom-6",
            "shoulder_link_geom",
            "elbow_link_geom",
            "forearm_link_geom",
            "wrist_link_geom",
            "wrist_hole_geom",
            "gripper_link_geom",
            "ar_tag_geom",
            "gripper_hole_geom",
            "finger_r_geom",
            "finger_l_geom",
        }
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.locobot_ik = AnalyticInverseKinematics()

        self._objects = ["object0", "object1", "object2"]

        # workspace boundaries for eef
        self._ws_low = [0.2, -0.17, -float('inf')]
        self._ws_high = [0.47, 0.17, float('inf')]

        self.set_opencv_camera_pose("main_cam", CAMERA_CALIB)


    def get_robot_mask(self, width=None, height=None):
        """
        Return binary img mask where 1 = robot and 0 = world pixel.
        robot_mask_with_obj means the robot mask is computed with object occlusions.
        """
        # returns a binary mask where robot pixels are True
        seg = self.render(
            "rgb_array", segmentation=True, width=width, height=height
        )  # flip the camera
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == self.mj_const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        if width is None or height is None:
            mask_dim = [self._img_height, self._img_width]
        else:
            mask_dim = [height, width]
        mask = np.zeros(mask_dim, dtype=np.bool)
        # TODO: change these to include the robot base
        # ignore_parts = {"finger_r_geom", "finger_l_geom"}

        ignore_parts = {}
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if name is not None:
                if name in ignore_parts:
                    continue
                if name in self._geoms:
                    mask[ids == i] = True
        return mask

    def get_gripper_pos(self, qpos):
        self.sim.data.qpos[self._joint_references] = qpos
        self.sim.forward()
        return self.sim.data.get_body_xpos("gripper_link").copy()

    def generate_masks(self, qpos_data, width=None, height=None):
        joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]
        finger_references = [
            self.sim.model.get_joint_qpos_addr(x) for x in ["joint_6", "joint_7"]
        ]
        masks = []
        for qpos in qpos_data:
            self.sim.data.qpos[joint_references] = qpos
            self.sim.data.qpos[finger_references] = [-0.025, 0.025]
            self.sim.forward()
            mask = self.get_robot_mask(width, height)
            masks.append(mask)
        masks = np.asarray(masks, dtype=np.bool)
        return masks

    def reset(self):
        reset_mocap2body_xpos(self.sim)
        reset_mocap_welds(self.sim)
        # first move the arm above to avoid object collision
        robot_above_qpos = [0.0, 0.43050715, 0.2393125, 0.63018035, 0.0]
        self.sim.data.qpos[self._joint_references] = robot_above_qpos
        self.sim.forward()
        # then sample object initialization
        self._sample_objects()
        # then bring robot gripper down
        curr_qpos = self.sim.data.qpos[self._joint_references][:4].copy()
        eef_target_pos = [0.27, 0.0, 0.18]
        # some noise to the x/y of the eef initial pos
        noise = np.random.uniform(-0.03, 0.03, size = 2)
        eef_target_pos[:2] += noise
        qpos_from_eef = np.zeros(5)
        qpos_from_eef[0:4] = self.locobot_ik.ik(
            eef_target_pos, alpha=-DEFAULT_PITCH, cur_arm_config=curr_qpos
        )
        qpos_from_eef[4] = DEFAULT_ROLL
        # qpos_from_eef = [0, 0.43050715, 0.2393125, 0.63018035, 0]
        # move robot to start pos
        self.sim.data.qpos[self._joint_references] = qpos_from_eef
        self.sim.forward()
        # print("qpos", qpos_from_eef)
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # check if applying action will violate the workspace boundary, if so, clip it.
        curr_eef_state = self.sim.data.get_body_xpos("finger_r").copy()
        next_eef_state = curr_eef_state + (action[:3] * 0.05)

        next_eef_state = np.clip(next_eef_state, self._ws_low, self._ws_high)
        clipped_ac = (next_eef_state - curr_eef_state) / 0.05
        action[:3] = clipped_ac


        self._set_action(clipped_ac)
        # gravity compensation
        self.sim.data.qfrc_applied[
            self._joint_vel_references
        ] = self.sim.data.qfrc_bias[self._joint_vel_references]
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {}
        reward = 0
        info["reward"] = reward
        return obs, reward, done, info

    def _set_action(self, action):
        # TODO: set joint action from end effector action using IK
        # use mocap to do it? since gripper position is in world coordinates
        assert action.shape == (3,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl = action[:3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        # default_rot = Quaternion(self.sim.data.mocap_quat[0].copy())
        # y_rot = Quaternion(axis=[1, 0, 0], degrees=10) # Rotate 5 deg about X
        # rot_ctrl = list(default_rot * y_rot)

        action = np.concatenate([pos_ctrl, rot_ctrl])
        mocap_set_action(self.sim, action)

    def _get_obs(self):
        """
        Return image, mask, robot state
        """
        if not hasattr(self, "_joint_references"):
            self._joint_references = [
                self.sim.model.get_joint_qpos_addr(x) for x in self._joints
            ]
            self._joint_vel_references = [
                self.sim.model.get_joint_qvel_addr(x) for x in self._joints
            ]

        img = self.render("rgb_array")
        masks = self.get_robot_mask()
        gripper_xpos = self.sim.data.get_body_xpos("finger_r").copy()
        # assume 0 for rotation, gripper force
        states = np.array([*gripper_xpos, 0, 0])
        qpos = self.sim.data.qpos[self._joint_references].copy()
        return {"observation": img, "masks": masks, "states": states, "qpos": qpos}

    def render(self, mode="rgb_array", segmentation=False, width=None, height=None):
        if width is None or height is None:
            width, height = self._img_width, self._img_height
        if mode == "rgb_array":
            data = self.sim.render(
                width,
                height,
                camera_name="main_cam",
                segmentation=segmentation,
                device_id=self._render_device,
            )
            # original image is upside-down, so flip it
            return data
        elif mode == "human":
            self._get_viewer(mode).render()

    def _sample_objects(self):
        # set objects in radius around spawn
        center = self.sim.data.get_site_xpos("spawn")[:2]
        spawn_id = self.sim.model.site_name2id("spawn")
        radius = self.sim.model.site_size[spawn_id][0]
        failed = False
        sampled_points = []
        for obj in self._objects:
            # reject sample if it overlaps with previous objects
            # reject sample if it's too close to the spawn point where the robot is
            for i in range(1000):
                no_overlap = True
                xy = self._sample_from_circle(center, radius)
                if np.linalg.norm(xy - center) < 0.08:
                    continue

                for other_xy in sampled_points:
                    if np.linalg.norm(xy - other_xy) < 0.07:
                        no_overlap = False
                        break
                if no_overlap:
                    sampled_points.append(xy)
                    break
            joint = obj + ":joint"
            pose = self.sim.data.get_joint_qpos(joint)
            z = pose[2]
            if no_overlap:
                obj_quat = [1,0,0,0]
                obj_pose = [xy[0], xy[1], z, *obj_quat]
                self.sim.data.set_joint_qpos(joint, obj_pose)
            else:
                failed = True
        # use default qpose if failed
        if failed:
            print("using default qpose since sampling failed")

    def _sample_from_circle(self, center, radius):
        """
        https://stackoverflow.com/questions/30564015/how-to-generate-random-points-in-a-circular-distribution
        """
        alpha = 2 * 3.1415 * np.random.uniform()
        r = radius * np.sqrt(np.random.uniform())
        # calculating coordinates
        x = r * np.cos(alpha) + center[0]
        y = r * np.sin(alpha) + center[1]
        return np.array([x, y])

    def _move(
        self,
        target,
        history,
        target_type="gripper",
        max_time=100,
        threshold=0.01,
        speed=10,
        noise=0,
    ):
        if target_type == "gripper":
            gripper_xpos = self.sim.data.get_body_xpos("finger_r").copy()
            d = target - gripper_xpos
        elif "object" in target_type:
            object_xpos = self.sim.data.get_site_xpos(target_type).copy()
            d = target - object_xpos
        step = 0
        while np.linalg.norm(d) > threshold and step < max_time:
            # add some random noise to ac
            if noise > 0:
                d[:2] = d[:2] + np.random.uniform(-noise, noise, size=2)
            ac = np.clip(d[:2] * speed, -1, 1)
            pad_ac = [*ac, 0]
            history["ac"].append(pad_ac)
            obs, _, _, info = self.step(pad_ac)
            history["obs"].append(obs)
            for k, v in info.items():
                history[k].append(v)
            if target_type == "gripper":
                gripper_xpos = self.sim.data.get_body_xpos("finger_r").copy()
                d = target - gripper_xpos
            elif "object" in target_type:
                object_xpos = self.sim.data.get_site_xpos(target_type).copy()
                d = target - object_xpos
            step += 1

    def generate_demo(self, behavior):
        """
        Runs a hard coded behavior and stores the episode
        Returns a dictionary with observation, action
        """
        self._behavior = behavior
        obs = self.reset()
        # self.render("human")
        history = defaultdict(list)
        history["obs"].append(obs)
        ep_len = self._config.demo_length
        beta = self._config.temporal_beta
        if behavior == "temporal_random_robot":
            self.temporal_random_robot(history, ep_len, beta)
        else:
            raise ValueError(behavior)

        return history

    def temporal_random_robot(self, history, ep_len, beta=1):
        """
        first moves robot near a random object, then
        generate temporally correlated actions
        """
        obj = np.random.choice(self._objects)
        history["pushed_obj"] = obj
        # move gripper behind the block and oriented for a goal push
        block_xpos = self.sim.data.get_site_xpos(obj).copy()
        spawn_xpos = self.sim.data.get_site_xpos("spawn").copy()
        goal_dir = (block_xpos - spawn_xpos) / np.linalg.norm(block_xpos - spawn_xpos)
        gripper_target = block_xpos - 0.05 * goal_dir
        # move robot near an object, record the actions
        self._move(gripper_target, history, speed=100, max_time=3)
        past_acs = len(history["ac"])
        # generate temporally correlated noise
        u = np.zeros((ep_len - 1, *self.action_space.shape))
        actions = np.zeros_like(u)
        actions[:past_acs] = history["ac"]
        for i in range(past_acs, ep_len - 1):
            u[i] = self.action_space.sample()
            actions[i] = beta * u[i] + (1 - beta) * actions[i - 1]
        history["ac"] = actions

        for i in range(past_acs, ep_len - 1):
            obs, _, _, info = self.step(actions[i])
            history["obs"].append(obs)
            for k, v in info.items():
                history[k].append(v)


if __name__ == "__main__":
    import sys
    import imageio
    from src.config import argparser
    from src.utils.mujoco import init_mjrender_device

    config, _ = argparser()
    init_mjrender_device(config)
    config.gpu = 0
    config.modified = True


    env = LocobotTableEnv(config)
    history = env.generate_demo("temporal_random_robot")
    gif = []
    for o in history["obs"]:
        img = o["observation"]
        mask = o["masks"]
        img[mask] = (0, 255, 255)
        gif.append(img)
    imageio.mimwrite("test.gif", gif)

    sys.exit(0)
    # env.get_robot_mask()
    # try locobot analytical ik
    # while True:
        # x, y, z = np.random.uniform(low=-1, high=1, size=3)
        # x, y = 0, 0
        # print(x,y,z)
        # env.render("human")
        # obs = env.step([x, y, 0])
    # env.render("human")
    while True:
        env.reset()
        env.render("human")
        for i in range(10):
            x,y,z = [1,0,0]
            obs = env.step([x, y, z])
            # print(env.sim.data.get_body_xpos("finger_r").copy()[0])
            env.render("human")
