import os
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces, utils
from mujoco_py.generated import const
from PIL import Image, ImageFilter
from skimage.filters import gaussian
from src.env.fetch.fetch_env import FetchEnv
from src.env.fetch.rotations import mat2euler
from src.env.fetch.utils import reset_mocap2body_xpos, reset_mocap_welds, robot_get_obs
import pickle

MODEL_XML_PATH = os.path.join("fetch", "push.xml")
LARGE_MODEL_XML_PATH = os.path.join("fetch", "large_push.xml")
INVISIBLE_LARGE_MODEL_XML_PATH = os.path.join("fetch", "invisible_large_push.xml")


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
            "object0:joint": [1, 0.75, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self._robot_pixel_weight = config.robot_pixel_weight
        reward_type = config.reward_type
        self._img_dim = config.img_dim
        self._camera_name = config.camera_name
        self._multiview = config.multiview
        self._camera_ids = config.camera_ids
        self._pixels_ob = config.pixels_ob
        self._depth_ob = config.depth_ob
        self._norobot_pixels_ob = config.norobot_pixels_ob
        self._distance_threshold = {
            "object": config.object_dist_threshold,
            "gripper": config.gripper_dist_threshold,
        }
        self._robot_goal_distribution = config.robot_goal_distribution
        self._push_dist = config.push_dist
        self._background_img = None
        self._large_block = config.large_block
        self._invisible_demo = config.invisible_demo
        xml_path = MODEL_XML_PATH
        if self._invisible_demo:
            xml_path = INVISIBLE_LARGE_MODEL_XML_PATH
        elif self._large_block:
            xml_path = LARGE_MODEL_XML_PATH

        self._blur_width = self._img_dim * 2
        self._sigma = config.blur_sigma
        self._unblur_cost_scale = config.unblur_cost_scale
        FetchEnv.__init__(
            self,
            xml_path,
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
            seed=config.seed,
        )
        utils.EzPickle.__init__(self)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype="float32")
        obs = self._get_obs()
        if self._pixels_ob:
            ob_dict = dict(
                desired_goal=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["achieved_goal"].shape,
                    dtype=np.uint8,
                ),
                achieved_goal=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["achieved_goal"].shape,
                    dtype=np.uint8,
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype=np.uint8
                ),
                robot=spaces.Box(-np.inf, np.inf, shape=(6,), dtype="float32"),
            )
            if self._depth_ob:
                ob_dict["world_coord"] = spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["world_coord"].shape,
                    dtype="float32",
                )
            self.observation_space = spaces.Dict(ob_dict)

    def robot_kinematics(self, sim_state, action):
        """
        Calculates the forward kinematics of the robot state.
        Does not actually affect the mujoco env.
        sim_state: mujoco state
        action: end effector control
        """
        self.set_state(sim_state)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        next_robot = self.get_robot_state()
        next_sim_state = self.get_state()
        self.set_state(sim_state)
        return next_robot, next_sim_state

    def get_robot_state(self):
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot = np.concatenate([grip_pos, grip_velp])
        return robot

    def _get_obs(self):
        # gripper positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        grip_velr = self.sim.data.get_site_xvelr("robot0:grip") * dt
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt
        if self._pixels_ob:
            out = self.render(
                "rgb_array", remove_robot=self._norobot_pixels_ob, get_depth_map=self._depth_ob
            )
            if self._depth_ob:
                img, world_coord = out
            else:
                img = out
            robot = np.concatenate([grip_pos, grip_velp])
            obs = {
                "observation": img.copy(),
                "achieved_goal": img.copy(),
                "desired_goal": self.goal.copy(),
                "robot": robot.copy(),
            }
            if self._depth_ob:
                obs["world_coord"] = world_coord
            return obs
        # change to a scalar if the gripper is made symmetric
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

    def reset(self):
        obs = super().reset()
        self._use_unblur = False
        return obs

    def _sample_goal(self):
        noise = np.zeros(3)
        # pushing axis noise
        # noise[0] = self.np_random.uniform(0.15, 0.15 + self.target_range, size=1)
        noise[0] = self._push_dist
        # side axis noise
        noise[1] = self.np_random.uniform(-0.02, 0.02, size=1)

        goal = self.initial_object_xpos[:3] + noise
        goal += self.target_offset
        goal[2] = self.height_offset

        init_state = self.get_state()
        # move block to target position
        obj_pose = [0, 0, 0, 1, 0, 0, 0]
        obj_pose[:3] = goal[:3]
        self.sim.data.set_joint_qpos("object0:joint", obj_pose)
        reset_mocap_welds(self.sim)
        self.sim.forward()
        # move robot behind block position or make it random
        obj_pos = self.sim.data.get_site_xpos("object0").copy()
        if self._robot_goal_distribution == "random":
            robot_noise = np.array([-0.1, 0, 0])  # 10cm behind block so no collide
            robot_noise[1] = self.np_random.uniform(-0.2, 0.2, size=1)  # side axis
            robot_noise[2] = self.np_random.uniform(0.01, 0.3, size=1)  # z axis
            gripper_target = obj_pos + robot_noise
        elif self._robot_goal_distribution == "behind_block":
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
        if self.reward_type in ["inpaint-blur", "inpaint", "weighted", "blackrobot"]:
            self.goal_mask = self.get_robot_mask()
            if self.reward_type in ["inpaint-blur", "inpaint"]:
                # inpaint the goal image with robot pixels
                goal[self.goal_mask] = self._background_img[self.goal_mask]
                self._unblurred_goal = goal
            elif self.reward_type == "blackrobot":
                # set the robot pixels to 0
                goal[self.goal_mask] = np.zeros(3)

        if self.reward_type == "inpaint-blur":
            # https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
            s = self._sigma
            w = self._blur_width
            t = (((w - 1) / 2) - 0.5) / s
            self._unblurred_goal = goal
            goal = np.uint8(
                255
                * gaussian(goal, sigma=s, truncate=t, mode="nearest", multichannel=True)
            )

        # reset to previous state
        self.set_state(init_state)
        reset_mocap2body_xpos(self.sim)
        reset_mocap_welds(self.sim)
        return goal

    def render(
        self,
        mode="rgb_array",
        camera_name=None,
        segmentation=False,
        remove_robot=False,
        get_depth_map=False,
    ):
        """
        If remove_robot, then use inpaint and mask to remove robot pixels
        remove_robot is used during dataset generation
        """
        import math
        from src.env.fetch.inverse_transform import pixel_coord_np, getHomogenousT
        from scipy.spatial.transform import Rotation as R
        if remove_robot:
            img = self.render()
            seg_mask = self.get_robot_mask()
            # inpaint the img
            img[seg_mask] = self._background_img[seg_mask]
            return img

        if self._multiview:
            imgs = []
            mv_world_coords = []
            for cam_id in self._camera_ids:
                camera_name = self.sim.model.camera_id2name(cam_id)
                if not get_depth_map:
                    img = super().render(
                        mode,
                        self._img_dim,
                        self._img_dim,
                        camera_name=camera_name,
                        segmentation=segmentation,
                    )
                else:
                    img, depth = self.sim.render(
                        width=self._img_dim,
                        height=self._img_dim,
                        camera_name=camera_name,
                        depth=True,
                    )
                    img, depth = img[::-1, :, :], depth[::-1, :]
                    extent = self.sim.model.stat.extent
                    near_ = self.sim.model.vis.map.znear * extent
                    far_ = self.sim.model.vis.map.zfar * extent

                    # intrinsics
                    height = width = self._img_dim
                    fovy = self.sim.model.cam_fovy[cam_id]
                    f = 0.5 * height / math.tan(fovy * math.pi / 360)
                    K = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
                    K_inv = np.linalg.inv(K)
                    depth = -(
                        near_ / (1 - depth * (1 - near_ / far_))
                    )  # -1 because camera is looking along the -Z axis of its frame
                    """
                    replace pixel coords with keypoints coordinates in pixel space
                    shape = (3,N) where N is no. of keypoints and third row is filled with 1s
                    """
                    pixel_coords = pixel_coord_np(width=width, height=height)
                    cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()
                    cam_quat = self.sim.model.cam_quat[cam_id]
                    cam_pos = self.sim.model.cam_pos[cam_id]
                    r = R.from_quat(
                        [cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]]
                    )
                    T = getHomogenousT(r.as_matrix(), cam_pos)
                    cam_homogenous_coords = np.vstack(
                        (cam_coords, np.ones(cam_coords.shape[1]))
                    )
                    world_coords = T @ cam_homogenous_coords
                    world_coords[:3, :] = world_coords[:3, :] / world_coords[-1, :]
                    mv_world_coords.append(world_coords[:3].reshape((3, height, width)))
                imgs.append(img)

            multiview_img = np.concatenate(imgs, axis=0)
            if get_depth_map:
                multiview_coords = np.concatenate(mv_world_coords, axis=1)
                return multiview_img, multiview_coords
            return multiview_img
        return super().render(
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
        for k, v in current_pose.items():
            dist = np.linalg.norm(v - self.goal_pose[k])
            info[f"{k}_dist"] = dist
            succ = dist < self._distance_threshold[k]
            info[f"{k}_success"] = float(succ)
        if self._robot_goal_distribution == "random":
            return info["object_success"]
        elif self._robot_goal_distribution == "behind_block":
            return float(info["object_success"] and info["gripper_success"])

    def weighted_cost(self, achieved_goal, goal, info):
        """
        inpaint-blur:
            need use_unblur boolean to decide when to switch from blur to unblur
            cost.
        """
        a = self._robot_pixel_weight
        ag_mask = self.get_robot_mask()
        if self.reward_type in ["inpaint", "inpaint-blur"]:
            # set robot pixels to background image
            achieved_goal[ag_mask] = self._background_img[ag_mask]
            if self.reward_type == "inpaint-blur":
                s = self._sigma
                w = self._blur_width
                t = (((w - 1) / 2) - 0.5) / s
                unblurred_ag = achieved_goal
                achieved_goal = np.uint8(
                    255
                    * gaussian(
                        achieved_goal,
                        sigma=s,
                        truncate=t,
                        mode="nearest",
                        multichannel=True,
                    )
                )
                blur_cost = np.linalg.norm(achieved_goal - goal)
                d = blur_cost
                if self._use_unblur:
                    unblur_cost = np.linalg.norm(unblurred_ag - self._unblurred_goal)
                    d = self._unblur_cost_scale * unblur_cost
                return -d

            else:
                pixel_costs = achieved_goal - goal
        elif self.reward_type == "weighted":
            # get costs per pixel
            pixel_costs = (achieved_goal - goal).astype(np.float64)
            pixel_costs[self.goal_mask] *= a
            pixel_costs[ag_mask] *= a
        elif self.reward_type == "blackrobot":
            # make robot black
            achieved_goal[ag_mask] = np.zeros(3)
            pixel_costs = achieved_goal - goal
        d = np.linalg.norm(pixel_costs)
        return -d

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == "none":
            return 0
        if self._pixels_ob:
            if self.reward_type in [
                "weighted",
                "inpaint",
                "inpaint-blur",
                "blackrobot",
            ]:
                return self.weighted_cost(achieved_goal, goal, info)
            # Compute distance between goal and the achieved goal.
            d = np.linalg.norm(achieved_goal - goal)
            if self.reward_type == "sparse":
                return -(d > self.distance_threshold).astype(np.float32)
            elif self.reward_type == "dense":
                return -d

        return super().compute_reward(achieved_goal, goal, info)

    def get_robot_mask(self):
        # returns a binary mask where robot pixels are True
        seg = self.render(segmentation=True)
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        mask_dim = [self._img_dim, self._img_dim]
        if self._multiview:
            viewpoints = len(self._camera_ids)
            mask_dim[0] *= viewpoints
        mask = np.zeros(mask_dim, dtype=np.uint8)
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if name is not None and "robot0:" in name:
                mask[ids == i] = np.ones(1, dtype=np.uint8)
        return mask.astype(bool)

    def _get_background_img(self):
        """
        Renders the background scene for the environment for inpainting
        Returns an image (H, W, C)
        """
        init_state = self.get_state()
        # move block to out of scene
        obj_pose = [100, 0, 0, 1, 0, 0, 0]
        self.sim.data.set_joint_qpos("object0:joint", obj_pose)
        reset_mocap_welds(self.sim)
        self.sim.forward()
        # move robot gripper up
        robot_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        robot_noise = np.array([-1, 0, 0.5])
        gripper_target = robot_pos + robot_noise

        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()
        self.sim.forward()
        img = self.render(mode="rgb_array")
        # reset to previous state
        self.set_state(init_state)
        reset_mocap2body_xpos(self.sim)
        reset_mocap_welds(self.sim)
        return img

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        if self._large_block:
            gripper_target = np.array(
                [-0.52 - 0.15, 0.005, -0.431 + self.gripper_extra_height]
            ) + self.sim.data.get_site_xpos("robot0:grip")
        else:
            gripper_target = np.array(
                [-0.498 - 0.15, 0.005, -0.431 + self.gripper_extra_height]
            ) + self.sim.data.get_site_xpos("robot0:grip")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        self.initial_object_xpos = self.sim.data.get_site_xpos("object0").copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos("object0")[2]

        if (
            self.reward_type in ["inpaint", "inpaint-blur"]
            and self._background_img is None
        ) or self._norobot_pixels_ob:
            self._background_img = self._get_background_img()

    def _set_action(self, action):
        assert action.shape == (3,)
        action = np.concatenate([action, [0]])
        super()._set_action(action)

    def _move(
        self,
        target,
        history,
        target_type="gripper",
        max_time=100,
        threshold=0.01,
        speed=10,
        noise=False,
    ):
        if target_type == "gripper":
            gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
            d = target - gripper_xpos
        elif target_type == "object":
            object_xpos = self.sim.data.get_site_xpos("object0").copy()
            d = target - object_xpos
        step = 0
        while np.linalg.norm(d) > threshold and step < max_time:
            # add some random noise to ac
            if noise:
                d += np.random.uniform(-0.05, 0.05, size=3)
            ac = d * speed
            history["ac"].append(ac)
            obs, _, _, info = self.step(ac)
            # self.render("human")
            history["obs"].append(obs)
            for k, v in info.items():
                history[k].append(v)
            self._vr.capture_frame()
            if self._record:
                history["frame"].append(self._vr.last_frame)
            if target_type == "gripper":
                gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
                d = target - gripper_xpos
            elif target_type == "object":
                object_xpos = self.sim.data.get_site_xpos("object0").copy()
                d = target - object_xpos
            step += 1

        # if np.linalg.norm(d) > threshold:
        #     print("move failed")
        # elif self._behavior == "push" and target_type == "object":
        #     goal_dist = np.linalg.norm(object_xpos - self.goal_pose["object"])
        #     print("goal object dist after push", goal_dist)

    def occlude(self, history):
        # move gripper above cube
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip")
        gripper_target = gripper_xpos + [0, 0, 0.048]
        self._move(gripper_target, history)
        # move gripper to occlude the cube
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip")
        gripper_target = gripper_xpos + [0.15, 0, 0]
        self._move(gripper_target, history)
        # move gripper downwards
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip")
        gripper_target = gripper_xpos + [0, 0, -0.061]
        self._move(gripper_target, history)

    def occlude_all(self, history):
        # move gripper above cube
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip")
        gripper_target = gripper_xpos + [0, 0, 0.05]
        self._move(gripper_target, history)
        # move gripper to occlude the cube
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip")
        gripper_target = gripper_xpos + [0.25, 0, 0]
        self._move(gripper_target, history, speed=10, threshold=0.025)
        # move gripper downwards
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip")
        gripper_target = gripper_xpos + [0, 0, -0.061]
        self._move(gripper_target, history, threshold=0.02)

    def push(self, history):
        # move gripper to center of cube
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        block_xpos = self.sim.data.get_site_xpos("object0").copy()
        gripper_target = gripper_xpos
        gripper_target[0] = block_xpos[0]
        self._move(gripper_target, history)
        # push the block
        obj_target = self.goal_pose["object"]
        if self._large_block:
            self._move(
                obj_target,
                history,
                target_type="object",
                speed=10,
                threshold=0.015,
                noise=True,
                max_time=13,
            )
        else:
            self._move(
                obj_target,
                history,
                target_type="object",
                speed=10,
                threshold=0.003,
            )

    def only_robot(self, history):
        # move gripper above cube
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip")
        gripper_target = gripper_xpos + [0, 0, 0.07]
        self._move(gripper_target, self._history)
        # move gripper to target robot pos
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip")
        gripper_target = self.goal_pose["gripper"]
        self._move(gripper_target, self._history, speed=10, threshold=0.025)

    def random_robot(self, history, ep_len):
        # randomly move the robot around
        for i in range(ep_len):
            ac = self.action_space.sample()
            history["ac"].append(ac)
            obs, _, _, info = self.step(ac)
            self._vr.capture_frame()
            history["obs"].append(obs)
            for k, v in info.items():
                history[k].append(v)

    def generate_demo(
        self, behavior, record=False, save_goal=False, record_path=None, ep_len=None
    ):
        """
        Behaviors: occlude, occlude_all, push, only robot move to goal
        """
        from collections import defaultdict

        from src.utils.video_recorder import VideoRecorder

        title_dict = {
            "weighted": f"Don't Care a={self._robot_pixel_weight}",
            "dense": "L2",
            "inpaint": "inpaint",
            "inpaint-blur": f"inpaint-blur_sig{self._sigma}",
            "blackrobot": "blackrobot",
        }
        size = "large" if self._large_block else "small"
        vp = "multi" if self._multiview else "single"
        if record_path is None:
            record_path = f"{size}_{behavior}_{vp}_view.mp4"
        self._vr = vr = VideoRecorder(self, path=record_path, enabled=record)
        self._record = record
        self._behavior = behavior
        obs = self.reset()
        # self.render("human")
        self._history = history = defaultdict(list)
        history["obs"].append(obs)
        vr.capture_frame()
        if record:
            history["frame"].append(vr.last_frame)
        history["goal"] = self.goal.copy()
        if save_goal:
            imageio.imwrite(
                f"{size}_{title_dict[self.reward_type]}_goal.png", history["goal"]
            )

        def rollout(history, path):
            frames = history["frame"]
            rewards = history["reward"]
            fig = plt.figure()
            rewards = -1 * np.array([0] + rewards)
            cols = len(frames)
            for n, (image, reward) in enumerate(zip(frames, rewards)):
                a = fig.add_subplot(2, cols, n + 1)
                imagegoal = np.concatenate([image, history["goal"]], axis=1)
                a.imshow(imagegoal)
                a.set_aspect("equal")
                # round reward to 2 decimals
                rew = f"{reward:0.2f}" if n > 0 else "Cost:"
                a.set_title(rew, fontsize=50)
                a.set_xticklabels([])
                a.set_xticks([])
                a.set_yticklabels([])
                a.set_yticks([])
                a.set_xlabel(f"step {n}", fontsize=40)
                # add goal img under every one
                # b = fig.add_subplot(2, cols, n + len(frames) + 1)
                # b.imshow(history["goal"])
                # b.set_aspect("equal")
                # obj =  f"{objd:0.3f}" if n > 0 else "Object Dist:"
                # b.set_title(obj, fontsize=50)
                # b.set_xticklabels([])
                # b.set_xticks([])
                # b.set_yticklabels([])
                # b.set_yticks([])
                # b.set_xlabel(f"goal", fontsize=40)

            fig.set_figheight(10)
            fig.set_figwidth(100)

            title = f"{title_dict[self.reward_type]} with {behavior} behavior"
            fig.suptitle(title, fontsize=50, fontweight="bold")
            fig.savefig(path)
            fig.clf()
            plt.close("all")

        if behavior == "occlude":
            self.occlude(history)
        elif behavior == "push":
            self.push(history)
        elif behavior == "occlude_all":
            self.occlude_all(history)
        elif behavior == "only_robot":
            self.only_robot(history)
        elif behavior == "random_robot":
            self.random_robot(history, ep_len)
        self._vr.close()
        # rollout(history, f"{title_dict[self.reward_type]}_{behavior}.png")
        return history


def plot_behaviors_per_cost():
    """ Plots a cost function's performance over behaviors"""
    config, _ = argparser()
    # visualize the initialization
    # cost_funcs = ["dontcare", "l2", "inpaint", "blackrobot", "alpha"]
    cost_funcs = ["inpaint-blur"]
    normalize = False
    data = {}
    viewpoints = ["single", "multiview"]
    behaviors = ["only_robot", "push", "occlude", "occlude_all"]
    # behaviors = ["push"]
    save_goal = False  # save a goal for each cost
    for behavior in behaviors:
        # only record once for each behavior
        record = False
        cost_traj = {}
        for cost in cost_funcs:
            cfg = deepcopy(config)
            if cost == "dontcare":
                cfg.reward_type = "weighted"
                cfg.robot_pixel_weight = 0
            elif cost == "l2":
                cfg.reward_type = "dense"
            elif cost == "inpaint":
                cfg.reward_type = "inpaint"
            elif cost == "inpaint-blur":
                cfg.reward_type = "inpaint-blur"
            elif cost == "blackrobot":
                cfg.reward_type = "blackrobot"
                cfg.robot_pixel_weight = 0
            elif cost == "alpha":
                # same as weighted but with alpha = 0.1
                cfg.reward_type = "weighted"
                cfg.robot_pixel_weight = 0.1

            cost_traj[cost] = {}
            for vp in viewpoints:
                cfg.multiview = vp == "multiview"
                env = FetchPushEnv(cfg)
                history = env.generate_demo(
                    behavior, record=record, save_goal=save_goal
                )
                cost_traj[cost][vp] = history
                env.close()
            record = False

        save_goal = False  # dont' need it for other behaviors
        data[behavior] = cost_traj

    if normalize:
        # get the min, max of costs across behaviors
        cost_min_dict = {vp: defaultdict(list) for vp in viewpoints}
        cost_max_dict = {vp: defaultdict(list) for vp in viewpoints}
        for behavior, cost_traj in data.items():
            for cost_fn, traj in cost_traj.items():
                for vp in viewpoints:
                    costs = -1 * np.array(traj[vp]["reward"])
                    cost_min_dict[vp][cost_fn].extend(costs)
                    cost_max_dict[vp][cost_fn].extend(costs)

    cmap = plt.get_cmap("Set1")
    for cost_fn in cost_funcs:
        for i, behavior in enumerate(behaviors):
            cost_traj = data[behavior][cost_fn]
            for vp in viewpoints:
                # graph the data
                size = "large" if cfg.large_block else "small"
                title = cost_fn
                if cost_fn == "inpaint-blur":
                    title = f"{cost_fn}-{env._sigma}"
                plt.title(f"{title} with {size} block")
                print(f"plotting {cost_fn} cost")
                costs = -1 * np.array(cost_traj[vp]["reward"])
                if normalize:
                    min = np.min(cost_min_dict[vp][cost_fn])
                    max = np.max(cost_max_dict[vp][cost_fn])
                    costs = (costs - min) / (max - min)

                timesteps = np.arange(len(costs)) + 1
                costs = np.array(costs)
                color = cmap(i)
                linestyle = "-" if vp == "multiview" else "--"
                plt.plot(
                    timesteps,
                    costs,
                    label=f"{behavior}_{vp[0]}",
                    linestyle=linestyle,
                    color=color,
                )

        plt.legend(loc="lower left", fontsize=9)
        plt.savefig(f"{size}_{cost_fn}_behaviors.png")
        plt.close("all")


def plot_costs_per_behavior():
    """ Plots the cost function for different behaviors"""
    config, _ = argparser()
    # visualize the initialization
    rewards = ["dontcare", "l2", "inpaint", "blackrobot", "alpha"]
    normalize = True
    data = {}
    behaviors = ["push", "occlude", "occlude_all", "only_robot"]
    save_goal = False  # save a goal for each cost
    for behavior in behaviors:
        # only record once for each behavior
        record = False
        cost_traj = {}
        for r in rewards:
            cfg = deepcopy(config)
            if r == "dontcare":
                cfg.reward_type = "weighted"
                cfg.robot_pixel_weight = 0
            elif r == "l2":
                cfg.reward_type = "dense"
            elif r == "inpaint":
                cfg.reward_type = "inpaint"
            elif r == "blackrobot":
                cfg.reward_type = "blackrobot"
                cfg.robot_pixel_weight = 0
            elif r == "alpha":
                # same as weighted but with alpha = 0.1
                cfg.reward_type = "weighted"
                cfg.robot_pixel_weight = 0.1

            env = FetchPushEnv(cfg)
            history = env.generate_demo(behavior, record=record, save_goal=save_goal)
            record = False
            cost_traj[r] = history
            env.close()

        save_goal = False  # dont' need it for other behaviors
        data[behavior] = cost_traj

    if normalize:
        # get the min, max of costs across behaviors
        cost_min_dict = defaultdict(list)
        cost_max_dict = defaultdict(list)
        for behavior, cost_traj in data.items():
            for cost_fn, traj in cost_traj.items():
                costs = -1 * np.array(traj["reward"])
                cost_min_dict[cost_fn].extend(costs)
                cost_max_dict[cost_fn].extend(costs)

    for behavior in behaviors:
        cost_traj = data[behavior]
        # graph the data
        size = "large" if cfg.large_block else "small"
        viewpoint = "multi" if cfg.multiview else "single"
        plt.title(f"Costs with {behavior} & {size} block & {viewpoint}-view")
        for cost_fn, traj in cost_traj.items():
            print(f"plotting {cost_fn} cost")
            costs = -1 * np.array(traj["reward"])
            if normalize:
                min = np.min(cost_min_dict[cost_fn])
                max = np.max(cost_max_dict[cost_fn])
                costs = (costs - min) / (max - min)
            timesteps = np.arange(len(costs)) + 1
            costs = np.array(costs)
            plt.plot(timesteps, costs, label=cost_fn)
        plt.legend(loc="upper right")
        plt.savefig(f"{size}_{behavior}_costs.png")
        plt.close("all")


def collect_trajectory(rank, config, behavior, record, num_trajectories, ep_len):
    config.seed = rank
    env = FetchPushEnv(config)
    len_stats = []
    it = range(num_trajectories)
    if rank == 0:
        it = tqdm(it)
    for i in it:
        # only record first episode for sanity check
        record = rank == 0 and i == 0 and record
        name = f"{behavior}_{rank}_{i}.hdf5"
        path = os.path.join(config.demo_dir, name)
        record_path = f"videos/{behavior}_{config.seed}_{i}.mp4"
        history = env.generate_demo(
            behavior, record=record, record_path=record_path, ep_len=ep_len
        )
        obs = history["obs"]  # array of observation dictionaries
        len_stats.append(len(obs))
        frames = []
        robot = []
        for ob in obs:
            frames.append(ob["observation"])
            robot.append(ob["robot"])

        frames = np.asarray(frames)
        robot = np.asarray(robot)
        actions = history["ac"]
        assert len(frames) - 1 == len(actions)
        with h5py.File(path, "w") as hf:
            hf.create_dataset("frames", data=frames, compression="gzip")
            hf.create_dataset("robot", data=robot, compression="gzip")
            hf.create_dataset("actions", data=actions, compression="gzip")

    # print out stats about the dataset
    stats_str = f"Avg len: {np.mean(len_stats)}\nstd: {np.std(len_stats)}\nmin: {np.min(len_stats)}\nmax: {np.max(len_stats)}"
    print(stats_str)
    stats_path = os.path.join(config.demo_dir, f"stats_{behavior}_{config.seed}.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)


def collect_trajectories():
    """
    Collect invisible robot block pushing
    """
    from multiprocessing import Process

    num_trajectories = 10  # per worker
    num_workers = 1
    record = False
    behavior = "push"
    ep_len = 12  # gonna be off by -1 because of reset but whatever

    config, _ = argparser()
    config.invisible_demo = False
    config.large_block = True
    config.demo_dir = "demos/fetch_push"
    os.makedirs(config.demo_dir, exist_ok=True)

    ps = []
    for i in range(num_workers):
        if i % 2 == 0:
            behavior = "random_robot"
        else:
            behavior = "push"
        p = Process(
            target=collect_trajectory,
            args=(i, config, behavior, record, num_trajectories, ep_len),
        )
        ps.append(p)

    for p in ps:
        p.start()

    for p in ps:
        p.join()


def collect_multiview_trajectory(
    rank, config, behavior, record, num_trajectories, ep_len
):
    # save the background image for inpainting?
    # save the robot segmentation mask?
    # or just save the inpainted image directly?
    config.seed = rank
    env = FetchPushEnv(config)
    len_stats = []
    it = range(num_trajectories)
    all_frames = []
    all_world_coord = []
    if rank == 0:
        it = tqdm(it)
    for i in it:
        # only record first episode for sanity check
        record = rank == 0 and i == 0 and record
        name = f"{behavior}_{rank}_{i}.hdf5"
        path = os.path.join(config.demo_dir, name)
        record_path = f"videos/{behavior}_{config.seed}_{i}.mp4"
        history = env.generate_demo(
            behavior, record=record, record_path=record_path, ep_len=ep_len
        )
        obs = history["obs"]  # array of observation dictionaries
        len_stats.append(len(obs))
        frames = []
        # robot = []
        world_coord = []
        for ob in obs:
            world_coord.append(ob["world_coord"].transpose(1,2,0))
            frames.append(ob["observation"])
            # robot.append(ob["robot"])

        frames = np.asarray(frames)
        world_coord = np.asarray(world_coord)

        all_world_coord.append(world_coord)
        all_frames.append(frames)
        # robot = np.asarray(robot)
        # actions = history["ac"]
        # assert len(frames) - 1 == len(actions)
    with h5py.File(path, "w") as hf:
        for i, (frame, world_coord) in tqdm(enumerate(zip(all_frames, all_world_coord))):
            hf.create_dataset(f"frame_{i}", data=frame, compression="gzip")
            hf.create_dataset(f"world_coord_{i}", data=world_coord, compression="gzip")
        # print("Frame shape:", all_frames.shape)
        # print("World Coord shape:", all_world_coord.shape)
        # hf.create_dataset("frames", data=all_frames, compression="gzip")
        # hf.create_dataset("world_coord", data=all_world_coord, compression="gzip")
        # hf.create_dataset("robot", data=robot, compression="gzip")
        # hf.create_dataset("actions", data=actions, compression="gzip")

    # print out stats about the dataset
    stats_str = f"Avg len: {np.mean(len_stats)}\nstd: {np.std(len_stats)}\nmin: {np.min(len_stats)}\nmax: {np.max(len_stats)}"
    print(stats_str)
    stats_path = os.path.join(config.demo_dir, f"stats_{behavior}_{config.seed}.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)


def collect_multiview_trajectories():
    """
    Collect multiview dataset with inpainting
    """
    from multiprocessing import Process

    num_trajectories = 2  # per worker
    num_workers = 1
    record = False
    behavior = "push"
    ep_len = 12  # gonna be off by -1 because of reset but whatever

    config, _ = argparser()
    config.large_block = True
    config.demo_dir = "demos/tckn_data"
    config.multiview = True
    config.norobot_pixels_ob = False
    config.reward_type = "dense"
    config.img_dim = 128
    config.depth_ob = True
    os.makedirs(config.demo_dir, exist_ok=True)

    if num_workers == 1:
        collect_multiview_trajectory(
            0, config, behavior, record, num_trajectories, ep_len
        )
    else:
        ps = []
        for i in range(num_workers):
            if i % 2 == 0:
                behavior = "random_robot"
            else:
                behavior = "push"
            p = Process(
                target=collect_multiview_trajectory,
                args=(i, config, behavior, record, num_trajectories, ep_len),
            )
            ps.append(p)

        for p in ps:
            p.start()

        for p in ps:
            p.join()


def collect_cem_goals():
    """Collect goal images for testing CEM planning"""
    config, _ = argparser()
    config.large_block = True
    config.demo_dir = "demos/cem_goals"
    config.multiview = True
    config.norobot_pixels_ob = True
    config.reward_type = "inpaint"
    config.img_dim = 64
    config.push_dist = 0.135  # with noise, (0.07, 0.2)
    os.makedirs(config.demo_dir, exist_ok=True)
    env = FetchPushEnv(config)
    for i in range(200):
        img = env._sample_goal()
        img_path = os.path.join(config.demo_dir, f"{i}.png")
        imageio.imwrite(img_path, img)


if __name__ == "__main__":
    from collections import defaultdict
    from copy import deepcopy

    import imageio
    from PIL import Image
    from src.config import argparser
    from torchvision.transforms import ToTensor
    import pickle

    # plot_behaviors_per_cost()
    # plot_costs_per_behavior()
    # collect_trajectories()
    collect_multiview_trajectories()
    # collect_cem_goals()
    # with open("demos/tckn_data/100push.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     import ipdb; ipdb.set_trace()
    # config, _ = argparser()
    # env = FetchPushEnv(config)
    # env.reset()
    # while True:
    #     env.step(env.action_space.sample())
    #     img = env.render("rgb_array").copy()
    #     tensor = ToTensor()(img)
    #     print(tensor.shape)
    #     import ipdb

    #     ipdb.set_trace()

    # img_width = 256
    # s = 5
    # w = 2 * img_width
    # t = (((w - 1) / 2) - 0.5) / s
    # goal = np.array(Image.open("code.png"))
    # goal = np.uint8(255 * gaussian(goal, sigma=s, truncate=t, mode="nearest", multichannel=True))
    # imageio.imwrite("blurredcode.png", goal)
