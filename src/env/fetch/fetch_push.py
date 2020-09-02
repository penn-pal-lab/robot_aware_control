import os

import numpy as np
from gym import utils

from src.env.fetch.fetch_env import FetchEnv
from src.env.fetch.utils import reset_mocap_welds, robot_get_obs
from src.env.fetch.rotations import mat2euler

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


class FetchPushEnv(FetchEnv, utils.EzPickle):
    """
    Pushes a block. We extend FetchEnv for:
    1) Pixel observations
    2) Image goal sampling where robot and block moves to goal location
    """
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            'robot0:slide0': 0.175,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.1,
            'object0:joint': [1.15, 0.75, 0.4, 1., 0., 0., 0.],
        }
        self._pixels_ob = False
        self._distance_threshold = {"object": 0.05, "gripper": 0.025}
        FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        if self._pixels_ob:
            obs = self.render("rgb_array")
            return {
                'observation': obs.copy(),
                'achieved_goal': obs.copy(),
                'desired_goal': self.goal.copy(),
            }
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        achieved_goal = np.concatenate([object_pos, grip_pos])
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
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
        noise[0] = self.np_random.uniform(0.15, 0.15 + self.target_range, size=1)
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
        obj_pos = self.sim.data.get_site_xpos('object0').copy()
        gripper_target = obj_pos + [-0.05, 0, 0]
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # set target site to obj pos
        site_id = self.sim.model.site_name2id('target0')
        sites_offset = (self.sim.data.site_xpos[site_id] - self.sim.model.site_pos[site_id]).copy()
        obj_pos = self.sim.data.get_site_xpos('object0').copy()
        self.sim.model.site_pos[site_id] = obj_pos - sites_offset
        self.sim.forward()
        obj_pos = self.sim.data.get_site_xpos('object0').copy()
        robot_pos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self._pixels_ob:
            goal = self.render(mode="rgb_array")
        else:
            goal = np.concatenate([obj_pos, robot_pos])

        # record goal pose for checking success later
        self.goal_pose = {
            "object": obj_pos,
            "gripper": robot_pos
        }
        # import imageio
        # imageio.imwrite(f'goal_{obj_pos[:3]}.png', goal)
        # reset to previous state
        self.set_state(init_state)
        return goal

    def render(self, mode='human', width=128, height=128):
        return super(FetchEnv, self).render(mode, width, height)

    def _render_callback(self):
        if self._pixels_ob:
            return
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
        self.sim.forward()

    def _is_success(self, achieved_goal, desired_goal):
        # if not self._pixels_ob:
        #     return super()._is_success(achieved_goal, desired_goal)
        current_pose = {
            "object": self.sim.data.get_site_xpos("object0").copy(),
            "gripper": self.sim.data.get_site_xpos('robot0:grip').copy()
        }
        success = True
        for k, v in current_pose.items():
            dist = np.linalg.norm(current_pose[k] - self.goal_pose[k])
            success &= dist < self._distance_threshold[k]
        return float(success)

    def compute_reward(self, achieved_goal, goal, info):
        # TODO: log failure in info dict
        if self._pixels_ob:
            # Compute distance between goal and the achieved goal.
            d = np.linalg.norm(achieved_goal - goal)
            if self.reward_type == 'sparse':
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -d
        return super().compute_reward(achieved_goal, goal, info)

if __name__ == "__main__":
    # visualize the initialization
    env = FetchPushEnv()
    while True:
        # env._sample_goal()
        env.reset()
        env.render()
