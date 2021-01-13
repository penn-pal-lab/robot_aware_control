import os
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
        # load the joint configuration and eef position
        path = "robonet_images/qposes_penn_baxter_left_traj14.npy"
        data = np.load(path)
        # set camera position
        # run qpos trajectory
        gif = []
        while True:
            for i, qpos in enumerate(data):
                self.sim.data.qpos[7:] = qpos
                self.sim.forward()
                img = self.render("rgb_array", width=320, height=240, camera_name="aux_cam")
                # self.render("human")
                real_img = imageio.imread(f"robonet_images/penn_baxter_left_traj14_{i}.png")
                comparison = np.concatenate([img, real_img], axis=1)
                gif.append(comparison)
            imageio.mimwrite("scene.gif", gif)
            break

    def _sample_goal(self):
        pass
    def _get_obs(self):
        return {"observation": np.array([0])}

if __name__ == "__main__":
    """
    press tab to cycle through cameras. There is the default camera, and the main_cam which we set the pose of in baxter/robot.xml
    """
    env = BaxterEnv()
