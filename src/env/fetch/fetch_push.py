import os

import numpy as np
from gym import utils
from mujoco_py.generated import const
import matplotlib.pyplot as plt

from src.env.fetch.fetch_env import FetchEnv
from src.env.fetch.utils import reset_mocap_welds, robot_get_obs, reset_mocap2body_xpos
from src.env.fetch.rotations import mat2euler

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "push.xml")


class FetchPushEnv(FetchEnv, utils.EzPickle):
    """
    Pushes a block. We extend FetchEnv for:
    1) Pixel observations
    2) Image goal sampling where robot and block moves to goal location
    3) reward_type: dense, weighted
    """

    def __init__(self, config):
        initial_qpos = {
            "robot0:slide0": 0.175,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.1,
            "object0:joint": [1.15, 0.75, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self._robot_pixel_weight = config.robot_pixel_weight
        reward_type = config.reward_type
        self._img_dim = config.img_dim
        # TODO: add static camera that is similar to original one
        self._camera_name = config.camera_name
        self._pixels_ob = config.pixels_ob
        self._distance_threshold = {
            "object": config.object_dist_threshold,
            "gripper": config.gripper_dist_threshold,
        }
        FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        if self._pixels_ob:
            obs = self.render("rgb_array")
            return {
                "observation": obs.copy(),
                "achieved_goal": obs.copy(),
                "desired_goal": self.goal.copy(),
            }
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos("object0")
            # rotations
            object_rot = mat2euler(self.sim.data.get_site_xmat("object0"))
            # velocities
            object_velp = self.sim.data.get_site_xvelp("object0") * dt
            object_velr = self.sim.data.get_site_xvelr("object0") * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric
        achieved_goal = np.concatenate([object_pos, grip_pos])
        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )
        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        # object_xpos = self.initial_gripper_xpos[:2]
        # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        #     object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        # object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        # assert object_qpos.shape == (7,)
        # object_qpos[:2] = object_xpos
        # self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        noise = np.zeros(3)
        # pushing axis noise
        # noise[0] = self.np_random.uniform(0.15, 0.15 + self.target_range, size=1)
        noise[0] = 0.25  # keep fixed for now
        # side axis noise
        noise[1] = self.np_random.uniform(-0.02, 0.02, size=1)

        goal = self.initial_object_xpos[:3] + noise
        goal += self.target_offset
        goal[2] = self.height_offset

        # if self._pixels_ob:
        init_state = self.get_state()
        # move block to target position
        obj_pose = [0, 0, 0, 1, 0, 0, 0]
        obj_pose[:3] = goal[:3]
        self.sim.data.set_joint_qpos("object0:joint", obj_pose)
        reset_mocap_welds(self.sim)
        self.sim.forward()
        # move robot behind block position
        obj_pos = self.sim.data.get_site_xpos("object0").copy()
        gripper_target = obj_pos + [-0.05, 0, 0]
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # set target site to obj pos
        site_id = self.sim.model.site_name2id("target0")
        sites_offset = (
            self.sim.data.site_xpos[site_id] - self.sim.model.site_pos[site_id]
        ).copy()
        obj_pos = self.sim.data.get_site_xpos("object0").copy()
        self.sim.model.site_pos[site_id] = obj_pos - sites_offset
        self.sim.forward()
        obj_pos = self.sim.data.get_site_xpos("object0").copy()
        robot_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        if self._pixels_ob:
            goal = self.render(mode="rgb_array")
        else:
            goal = np.concatenate([obj_pos, robot_pos])

        # record goal info for checking success later
        self.goal_pose = {"object": obj_pos, "gripper": robot_pos}
        if self.reward_type == "weighted":
            self.goal_mask = self.get_robot_mask()
            goal = (
                self._robot_pixel_weight * self.goal_mask * goal
                + (1 - self.goal_mask) * goal
            )

        # reset to previous state
        self.set_state(init_state)
        reset_mocap2body_xpos(self.sim)
        reset_mocap_welds(self.sim)
        return goal

    def render(
        self,
        mode="rgb_array",
        width=512,
        height=512,
        camera_name=None,
        segmentation=False,
    ):
        return super(FetchEnv, self).render(
            mode,
            self._img_dim,
            self._img_dim,
            camera_name=self._camera_name,
            segmentation=segmentation,
        )

    def _render_callback(self):
        return

    def _is_success(self, achieved_goal, desired_goal, info):
        current_pose = {
            "object": self.sim.data.get_site_xpos("object0").copy(),
            "gripper": self.sim.data.get_site_xpos("robot0:grip").copy(),
        }
        success = True
        for k, v in current_pose.items():
            dist = np.linalg.norm(current_pose[k] - self.goal_pose[k])
            info[f"{k}_dist"] = dist
            succ = dist < self._distance_threshold[k]
            info[f"{k}_success"] = float(succ)
            success &= succ
        return float(success)

    def weighted_cost(self, achieved_goal, goal, info):
        a = self._robot_pixel_weight
        ag_mask = self.get_robot_mask()
        robot_pixels = ag_mask * achieved_goal
        scaled_robot_pixels = a * robot_pixels
        non_robot_pixels = (1 - ag_mask) * achieved_goal
        reweighted_ag = scaled_robot_pixels + non_robot_pixels

        d = np.linalg.norm(reweighted_ag - goal)
        return -d

    def compute_reward(self, achieved_goal, goal, info):
        if self._pixels_ob:
            if self.reward_type == "weighted":
                return self.weighted_cost(achieved_goal, goal, info)
            # Compute distance between goal and the achieved goal.
            d = np.linalg.norm(achieved_goal - goal)
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32)
            elif self.reward_type == "dense":
                return -d

        return super().compute_reward(achieved_goal, goal, info)

    def get_robot_mask(self):
        seg = self.render(segmentation=True)
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        mask = np.zeros((self._img_dim, self._img_dim, 3), dtype=np.uint8)
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if name is not None and "robot0:" in name:
                mask[ids == i] = np.ones(3, dtype=np.uint8)
        return mask


if __name__ == "__main__":
    from src.config import argparser
    import imageio

    """
    If `segmentation` is True, this is a (height, width, 2) int32 numpy
          array where the second channel contains the integer ID of the object at
          each pixel, and the  first channel contains the corresponding object
          type (a value in the `mjtObj` enum). Background pixels are labeled
          (-1, -1).
    """
    config, _ = argparser()
    # visualize the initialization
    env = FetchPushEnv(config)
    img_dim = config.img_dim
    while True:
        # env._sample_goal()
        env.reset()
        # env.render(mode="human", segmentation=True)
        # continue
        img = env.render(segmentation=False)
        imageio.imwrite("scene.png", img.astype(np.uint8))
        seg = env.render(segmentation=True)
        # visualize all the bodies
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        robot_geoms = {
            "robot0:base_link",
            "robot0:torso_lift_link",
            "robot0:head_pan_link",
            "robot0:head_tilt_link",
            "robot0:shoulder_pan_link",
            "robot0:shoulder_lift_link",
            "robot0:upperarm_roll_link",
            "robot0:elbow_flex_link",
            "robot0:forearm_roll_link",
            "robot0:wrist_flex_link",
            "robot0:wrist_roll_link",
            "robot0:gripper_link",
            "robot0:r_gripper_finger_link",
            "robot0:l_gripper_finger_link",
            "robot0:estop_link",
            "robot0:laser_link",
            "robot0:torso_fixed_link",
        }
        img = np.zeros((img_dim, img_dim), dtype=np.uint8)
        # robot_color = np.array((255, 0, 0), dtype=np.uint8)
        robot_color = 255
        for i in geoms_ids:
            name = env.sim.model.geom_id2name(i)
            if name is not None and name in robot_geoms:
                img[ids == i] = robot_color
            # elif "object0" == name:
            #     img[ids == i] = np.array((0, 255, 0), dtype=np.uint8)
        img = img.astype(np.uint8)
        imageio.imwrite("seg.png", img)
        # from PIL import Image, ImageFilter
        # pil_img = Image.fromarray(img)
        # img_nn_pil = pil_img.filter(ImageFilter.BoxBlur(1))
        # imageio.imwrite('seg_nn.png', img_nn_pil)
        break
