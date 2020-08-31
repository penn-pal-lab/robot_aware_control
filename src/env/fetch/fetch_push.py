import os

import numpy as np
from gym import utils

from env.fetch.fetch_env import FetchEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


class FetchPushEnv(FetchEnv, utils.EzPickle):
    """Pushes a block
    """
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            'robot0:slide0': 0.175,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.1,
            'object0:joint': [1.15, 0.75, 0.4, 1., 0., 0., 0.],
        }
        FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

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
        noise[0] = self.np_random.uniform(0.2, 0.2 + self.target_range, size=1)
        # side axis noise
        noise[1] = self.np_random.uniform(-0.02, 0.02, size=1)

        goal = self.initial_object_xpos[:3] + noise
        goal += self.target_offset
        goal[2] = self.height_offset
        return goal.copy()


if __name__ == "__main__":
    # visualize the initialization
    env = FetchPushEnv()
    while True:
        env.reset()
        env.render()
