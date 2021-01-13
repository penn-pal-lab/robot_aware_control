import os
from src.env.fetch.rotations import euler2quat, quat2euler
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

    def compare_traj(self):
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
                mask = self.get_robot_mask()
                real_img = imageio.imread(f"robonet_images/penn_baxter_left_traj14_{i}.png")
                mask_img = real_img.copy()
                mask_img[mask] = 255
                # imageio.imwrite("mask_img.png", mask_img)
                # import ipdb; ipdb.set_trace()
                comparison = np.concatenate([img, real_img, mask_img], axis=1)
                gif.append(comparison)
            imageio.mimwrite("scene.gif", gif)
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
        seg = self.render("rgb_array", width=320, height=240, camera_name="aux_cam", segmentation=True)
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == self.mj_const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        mask_dim = [240, 320]
        mask = np.zeros(mask_dim, dtype=np.bool)
        ignore_parts = {"base_link_vis", "base_link_col","head_vis"}
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
    Compares the simulated scene, real scene, and robot mask
    """
    env = BaxterEnv()
    env.compare_traj()

    """
    Scene Visualization
    press tab to cycle through cameras. There is the default camera, and the main_cam which we set the pose of in baxter/robot.xml
    """
    # while True:
    #     env.render("human")
