import math
import os
from collections import defaultdict
from copy import deepcopy

import imageio

# import ipdb
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces, utils
from scipy.spatial.transform import Rotation as R
from skimage.filters import gaussian
from src.env.fetch.collision import CollisionSphere
from src.env.fetch.fetch_env import FetchEnv
from src.env.fetch.inverse_transform import getHomogenousT, pixel_coord_np
from src.env.fetch.planar_rrt import PlanarRRT
from src.env.fetch.rotations import mat2euler
from src.env.fetch.utils import reset_mocap2body_xpos, reset_mocap_welds, robot_get_obs

MODEL_XML_PATH = os.path.join("fetch", "clutterpush.xml")


class ClutterPushEnv(FetchEnv, utils.EzPickle):
    """
    Pushes a block. We extend FetchEnv for:
    1) Pixel observations
    2) Image goal sampling where robot and block moves to goal location
    3) reward_type: dense, weighted
    """

    def __init__(self, config):
        # initialize objects out of bounds first
        self.initial_qpos = initial_qpos = {
            "robot0:slide0": 0.175,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.1,
            "object0:joint": [3, 0.75, 0.42, 1.0, 0.0, 0.0, 0.0],
            "object1:joint": [3.1, 0.75, 0.44, 1.0, 0.0, 0.0, 0.0],
            "object2:joint": [3.2, 0.75, 0.44, 1.0, 0.0, 0.0, 0.0],
        }
        self._objects = ["object0", "object1", "object2"]
        self._robot_pixel_weight = config.robot_pixel_weight
        reward_type = config.reward_type
        self._img_dim = config.img_dim
        self._camera_name = config.camera_name
        self._multiview = config.multiview
        self._camera_ids = config.camera_ids
        self._pixels_ob = config.pixels_ob
        self._depth_ob = config.depth_ob
        self._norobot_pixels_ob = config.norobot_pixels_ob
        self._inpaint_eef = config.inpaint_eef
        self._distance_threshold = {
            o: config.object_dist_threshold for o in self._objects
        }
        self._robot_goal_distribution = config.robot_goal_distribution
        self._push_dist = config.push_dist
        self._background_img = None
        self._invisible_demo = config.invisible_demo
        xml_path = MODEL_XML_PATH
        self._blur_width = self._img_dim * 2
        self._sigma = config.blur_sigma
        self._unblur_cost_scale = config.unblur_cost_scale
        self._most_recent_background = config.most_recent_background
        self._action_repeat = config.action_repeat
        self._config = config
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
        utils.EzPickle.__init__(self, config)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype="float32")
        obs = self._get_obs()
        if self._pixels_ob:
            # self.observation_space = spaces.Dict(
            #     dict(
            #         desired_goal=spaces.Box(
            #             -np.inf,
            #             np.inf,
            #             shape=obs["achieved_goal"].shape,
            #             dtype=np.uint8,
            #         ),
            #         achieved_goal=spaces.Box(
            #             -np.inf,
            #             np.inf,
            #             shape=obs["achieved_goal"].shape,
            #             dtype=np.uint8,
            #         ),
            #         observation=spaces.Box(
            #             -np.inf, np.inf, shape=obs["observation"].shape, dtype=np.uint8
            #         ),
            #         robot=spaces.Box(-np.inf, np.inf, shape=(6,), dtype="float32"),
            #     )
            # )
            self.observation_space = spaces.Dict(
                dict(
                    observation=spaces.Box(
                        -np.inf, np.inf, shape=obs["observation"].shape, dtype=np.uint8
                    ),
                    robot=spaces.Box(-np.inf, np.inf, shape=(6,), dtype="float32"),
                )
            )

    def robot_kinematics(self, sim_state, action, ret_mask=False):
        """
        Calculates the forward kinematics of the robot state.
        Does not actually affect the mujoco env.
        sim_state: mujoco state
        action: end effector control
        """
        self.set_flattened_state(sim_state)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(self._action_repeat):
            self._set_action(action)
            self.sim.step()
            self._step_callback()
        next_robot = self.get_robot_state()
        if ret_mask:
            next_mask = self.get_robot_mask()
        next_sim_state = self.get_flattened_state()
        self.set_flattened_state(sim_state)
        if ret_mask:
            return next_robot, next_mask, next_sim_state
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
            img = self.render("rgb_array", remove_robot=self._norobot_pixels_ob, get_depth_map=self._depth_ob)
            if self._depth_ob:
                img, world_coord = img
            robot = np.concatenate([grip_pos, grip_velp])
            obs = {
                "observation": img.copy(),
                "robot": robot.copy().astype(np.float32),
                "state": self.get_flattened_state(),
            }
            for obj in self._objects:
                obs[obj + ":joint"] = self.sim.data.get_joint_qpos(
                    obj + ":joint"
                ).copy()
            if self._norobot_pixels_ob:
                obs["mask"] = self._seg_mask
            else:
                obs["mask"] = self.get_robot_mask()
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

    def _get_auto_moving_obj_obs(
        self, moving_object="object1", moving_dist=np.zeros(3)
    ):
        # gripper positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        grip_velr = self.sim.data.get_site_xvelr("robot0:grip") * dt
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt

        obj_pos = self.sim.data.get_joint_qpos(moving_object + ":joint")
        obj_pos[0:3] += moving_dist
        self.sim.data.set_joint_qpos(moving_object + ":joint", obj_pos)

        if self._pixels_ob:
            img = self.render("rgb_array", remove_robot=self._norobot_pixels_ob)
            robot = np.concatenate([grip_pos, grip_velp])
            obs = {
                "observation": img.copy(),
                # "achieved_goal": img.copy(),
                # "desired_goal": self.goal.copy(),
                "robot": robot.copy().astype(np.float32),
                "state": self.get_flattened_state(),
            }
            for obj in self._objects:
                obs[obj + ":joint"] = self.sim.data.get_joint_qpos(
                    obj + ":joint"
                ).copy()
            if self._norobot_pixels_ob:
                obs["mask"] = self._seg_mask
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

    def _reset_sim(self, spawn_info=None):
        """
        Gets called by goal env's reset right before sampling new goal.
        spawn_info will have the poses of the objects and the robot spawn point
        """
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        # object_xpos = self.initial_gripper_xpos[:2]
        # while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        #     object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        # object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        # assert object_qpos.shape == (7,)
        # object_qpos[:2] = object_xpos
        # self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        if spawn_info is not None:
            for obj in self._objects:
                obj_joint = obj + ":joint"
                obj_pose = self.sim.data.get_joint_qpos(obj_joint).copy()
                obj_pose = spawn_info[obj]
                self.sim.data.set_joint_qpos(obj_joint, obj_pose)
        else:
            self._sample_objects()
        self.sim.forward()
        return True

    def reset(self, init_state=None):
        """
        use init_state for initializing with demo
        """
        if init_state is not None:
            self.set_flattened_state(init_state)
        else:
            did_reset_sim = False
            while not did_reset_sim:
                did_reset_sim = self._reset_sim()

        if (
            self.reward_type in ["inpaint", "inpaint-blur"] or self._norobot_pixels_ob
        ) and self._most_recent_background:
            self._background_img = self._get_background_img()

        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        self._use_unblur = False
        return obs

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(self._action_repeat):
            self._set_action(action)
            self.sim.step()
            self._step_callback()
        obs = self._get_obs()
        done = False
        info = {}
        reward = 0
        info["reward"] = reward
        return obs, reward, done, info

    def step_auto_moving_obj(
        self, action, moving_object="object1", moving_dist=np.zeros(3)
    ):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for _ in range(self._action_repeat):
            self._set_action(action)
            self.sim.step()
            self._step_callback()
        obs = self._get_auto_moving_obj_obs(
            moving_object=moving_object, moving_dist=moving_dist
        )
        done = False
        info = {}
        reward = 0
        info["reward"] = reward
        return obs, reward, done, info

    def make_push_object_demo(self):
        """
        Generates a demonstration containing a sequence of images of the puck
        moving towards the goal.
        For each object:
            Move the object to the goal using straight push or RRT
            Update the obstacle list with the goal pose
        Returns the path, imgs, info
        info dict contains start and goal pos, order of parts pushed, and idx of each
        object's push trajectory
        """
        # move robot gripper up so it's not in view of camera
        init_state = self.get_state()
        robot_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
        gripper_target = robot_pos + np.array([-1, 0, 0.5])

        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()
        self.sim.forward()

        start_pos = {
            obj: self._get_joint_qpos(obj + ":joint")[:2] for obj in self._objects
        }
        goal_pos = {obj: pose[:2] for obj, pose in self.goal_pose.items()}
        collision_radius = 0.07
        obstacles = {
            k: CollisionSphere(p, collision_radius) for k, p in start_pos.items()
        }
        info = {}
        for k, v in start_pos.items():
            info["start_" + k] = v
        for k, v in goal_pos.items():
            info["goal_" + k] = v
        imgs = []
        all_path = []
        info["push_order"] = []
        push_order = np.random.permutation(len(self._objects))
        for idx in push_order:
            obj = self._objects[idx]
            # attempt straight line push
            start_state = start_pos[obj]
            goal_state = goal_pos[obj]
            if np.linalg.norm(start_state - goal_state) < 0.01:
                continue
            info["push_order"].append(idx)
            can_push_straight = True
            for name, collider in obstacles.items():
                if obj == name:
                    continue
                goal_start = goal_state - start_state
                u = (goal_start) / np.linalg.norm(goal_start)
                if collider.line_in_collision(start_state, u):
                    can_push_straight = False
                    break

            if can_push_straight:
                path = np.linspace(start_state, goal_state, num=5)
            else:
                # dimensions of arena
                center = self.sim.data.get_site_xpos("spawn")[:2]
                left = self.sim.data.get_site_xpos("arenaleft")
                right = self.sim.data.get_site_xpos("arenaright")
                top = self.sim.data.get_site_xpos("arenatop")
                bot = self.sim.data.get_site_xpos("arenabottom")
                height = np.abs(top - bot)[0] / 2
                width = np.abs(right - left)[1] / 2
                dim_ranges = [
                    (center[0] - height, center[0] + height),
                    (center[1] - width, center[1] + width),
                ]
                colliders = []
                for name, collider in obstacles.items():
                    if name != obj:
                        colliders.append(collider)
                rrt1 = PlanarRRT(
                    start_state=start_state,
                    goal_state=goal_state,
                    dim_ranges=dim_ranges,
                    max_iter=10000,
                    obstacles=colliders,
                    step_size=0.05,
                    visualize=False,
                    goal_bias=0.3,
                    visualize_path=f"{obj}.png",
                )

                path = rrt1.build()
                if path is None:
                    print("Could not build a path")

            info[obj + "_straight_push"] = can_push_straight
            info[obj + "_idx"] = (len(all_path), len(all_path) + len(path))
            all_path.extend(path)
            for p in path:
                obj_pose = self._get_joint_qpos(obj + ":joint")
                obj_pose = np.concatenate([p, obj_pose[2:]])
                self._set_joint_qpos(obj + ":joint", obj_pose)
                self.sim.forward()
                imgs.append(self.render("rgb_array"))
            # update object's collision to be after push
            obstacles[obj] = CollisionSphere(goal_state, collision_radius)

        self.set_state(init_state)
        reset_mocap2body_xpos(self.sim)
        reset_mocap_welds(self.sim)
        return all_path, imgs, info

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

    def _sample_from_rectangle(self, center, half_width, half_height):
        x = np.random.uniform(-half_width, half_width) + center[0]
        y = np.random.uniform(-half_height, half_height) + center[1]
        return np.array([x, y])

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
                if np.linalg.norm(xy - center) < 0.09:
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
                obj_quat = pose[3:]
                obj_pose = [xy[0], xy[1], z, *obj_quat]
                self.sim.data.set_joint_qpos(joint, obj_pose)
            else:
                failed = True
        # use default qpose if failed
        if failed:
            print("using default qpose since sampling failed")
            for k, v in self.initial_qpos.items():
                if "object" in k:
                    obj_pose = list(v)
                    obj_pose[2] = 0.43
                    self.sim.data.set_joint_qpos(k, obj_pose)

    def set_goal(self, goal_info):
        """
        Assume goal is image, and reward type is either L2, inpaint, inpaint-blur
        """
        # self.goal_pose contains goal poses of all objects
        # self.current_goal_obj is the string of the current object to push

        # self.goal_pose[obj][:2] = goal_state
        # goal = self._unblurred_goal = img
        # if self.reward_type == "inpaint-blur":
        #     # https://stackoverflow.com/questions/25216382/gaussian-filter-in-scipy
        #     s = self._sigma
        #     w = self._blur_width
        #     t = (((w - 1) / 2) - 0.5) / s
        #     self._unblurred_goal = goal
        #     goal = np.uint8(
        #         255
        #         * gaussian(goal, sigma=s, truncate=t, mode="nearest", multichannel=True)
        #     )
        # return goal

    def _sample_goal(self):
        """
        For each object, place it somewhere in the arena
        reject sample if it is in spawn or overlapping with another object either in
        spawn or goal
        """
        init_state = self.get_state()
        # first turn off robot collision
        robot_col = {}
        for geom_id, _ in enumerate(self.sim.model.geom_bodyid):
            geom_name = self.sim.model.geom_id2name(geom_id)
            if geom_name is not None and "robot" in geom_name:
                robot_col[geom_name] = (
                    self.sim.model.geom_contype[geom_id],
                    self.sim.model.geom_conaffinity[geom_id],
                )
                self.sim.model.geom_contype[geom_id] = 0
                self.sim.model.geom_conaffinity[geom_id] = 0

        # dimensions of arena
        center = self.sim.data.get_site_xpos("spawn")[:2]
        left = self.sim.data.get_site_xpos("arenaleft")
        right = self.sim.data.get_site_xpos("arenaright")
        top = self.sim.data.get_site_xpos("arenatop")
        bot = self.sim.data.get_site_xpos("arenabottom")
        width = np.abs(right - left)[1] / 2
        height = np.abs(top - bot)[0] / 2
        # dimensions of spawn
        spawn_id = self.sim.model.site_name2id("spawn")
        radius = self.sim.model.site_size[spawn_id][0]
        failed = False
        sampled_points = []
        for obj in self._objects:  # add spawn points to consider for overlap
            spawn_xy = self._get_joint_qpos(obj + ":joint")[:2]
            sampled_points.append(spawn_xy)

        goal_order = np.random.permutation(self._objects)
        # only choose 1 object to push for easy task
        for obj in goal_order[:1]:
            # reject sample if it overlaps with previous objects
            for i in range(1000):
                no_overlap = True
                xy = self._sample_from_rectangle(center, width, height)
                for other_xy in sampled_points:
                    if np.linalg.norm(xy - other_xy) < 0.08 or np.linalg.norm(
                        xy - center
                    ) < (radius + 0.06):
                        no_overlap = False
                        break
                if no_overlap:
                    sampled_points.append(xy)
                    break
            joint = obj + ":joint"
            pose = self._get_joint_qpos(joint)
            z = pose[2]
            if no_overlap:
                obj_quat = [1, 0, 0, 0]
                obj_pose = [xy[0], xy[1], z, *obj_quat]
                self._set_joint_qpos(joint, obj_pose)
            else:
                failed = True
        # use default qpose if failed
        if failed:
            print("using default qpose since sampling failed")
            for k, v in self.initial_qpos.items():
                if "object" in k:
                    obj_pose = list(v)
                    self._set_joint_qpos(k, obj_pose)
        self.sim.forward()
        # move robot arm back to the center
        # obj_pos = self.sim.data.get_site_xpos("object0").copy()
        # if self._robot_goal_distribution == "random":
        #     robot_noise = np.array([-0.1, 0, 0])  # 10cm behind block so no collide
        #     robot_noise[1] = self.np_random.uniform(-0.2, 0.2, size=1)  # side axis
        #     robot_noise[2] = self.np_random.uniform(0.01, 0.3, size=1)  # z axis
        #     gripper_target = obj_pos + robot_noise
        # elif self._robot_goal_distribution == "behind_block":
        #     gripper_target = obj_pos + [-0.05, 0, 0]
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        gripper_target = [0.8, 0.75, 0.9]
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # reenable robot collision
        for geom_id, _ in enumerate(self.sim.model.geom_bodyid):
            geom_name = self.sim.model.geom_id2name(geom_id)
            if geom_name is not None and "robot" in geom_name:
                contype, conaffinity = robot_col[geom_name]
                self.sim.model.geom_contype[geom_id] = contype
                self.sim.model.geom_conaffinity[geom_id] = conaffinity
        self.sim.step()

        if self._pixels_ob:
            goal = self.render(mode="rgb_array")
        else:
            obj_pos = self.sim.data.get_site_xpos("object0").copy()
            robot_pos = self.sim.data.get_site_xpos("robot0:grip").copy()
            goal = np.concatenate([obj_pos, robot_pos])

        # record goal info for checking success later
        self.goal_pose = {
            obj: self._get_joint_qpos(obj + ":joint")[:3] for obj in self._objects
        }
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
        if remove_robot:
            img = self.render()
            seg_mask = self._seg_mask = self.get_robot_mask()
            if self._most_recent_background:
                # update background img to most recent unoccluded pixels.
                # this is not useful for the current timestep, but will be useful
                # for future inpaintings
                self._background_img[~seg_mask] = img[~seg_mask].copy()
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
            obj: self._get_joint_qpos(obj + ":joint")[:2] for obj in self._objects
        }
        obj_successes = []
        for k, v in self.goal_pose.items():
            dist = np.linalg.norm(current_pose[k] - v[:2])
            info[f"{k}_dist"] = dist
            succ = dist < self._distance_threshold[k]
            info[f"{k}_success"] = float(succ)
            obj_successes.append(succ)
        return all(obj_successes)
        # if self._robot_goal_distribution == "random":
        #     return info["object_success"]
        # elif self._robot_goal_distribution == "behind_block":
        #     return float(info["object_success"] and info["gripper_success"])

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
                if self._use_unblur:
                    unblurred_ag = achieved_goal
                    unblur_cost = np.linalg.norm(unblurred_ag - self._unblurred_goal)
                    d = self._unblur_cost_scale * unblur_cost
                else:
                    s = self._sigma
                    w = self._blur_width
                    t = (((w - 1) / 2) - 0.5) / s
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
        geoms = types == self.mj_const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        eef_geoms = ["robot0:r_gripper_finger_link", "robot0:l_gripper_finger_link"]
        mask_dim = [self._img_dim, self._img_dim]
        if self._multiview:
            viewpoints = len(self._camera_ids)
            mask_dim[0] *= viewpoints
        mask = np.zeros(mask_dim, dtype=np.bool)
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if not self._inpaint_eef and name in eef_geoms:
                continue
            if name is not None and "robot0:" in name:
                mask[ids == i] = True
        return mask

    def _get_background_img(self):
        """
        Renders the background scene for the environment for inpainting
        Returns an image (H, W, C)
        """
        init_state = self.get_state()
        reset_mocap_welds(self.sim)
        self.sim.forward()
        # move entire robot out of sight
        self.sim.data.set_joint_qpos("robot0:slide2", 1)
        self.sim.data.set_joint_qpos("robot0:slide0", -1)
        # move robot gripper to above spawn
        # gripper_target = self.sim.data.get_site_xpos("spawn") + np.array([0, 0, 0.5])

        # gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        # self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        # self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        # for _ in range(10):
        #     self.sim.step()
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
        # first disable robot collision
        robot_col = {}
        for geom_id, _ in enumerate(self.sim.model.geom_bodyid):
            geom_name = self.sim.model.geom_id2name(geom_id)
            if geom_name is not None and "robot" in geom_name:
                robot_col[geom_name] = (
                    self.sim.model.geom_contype[geom_id],
                    self.sim.model.geom_conaffinity[geom_id],
                )
                self.sim.model.geom_contype[geom_id] = 0
                self.sim.model.geom_conaffinity[geom_id] = 0
        gripper_target = np.array([0, 0, 0.05]) + self.sim.data.get_site_xpos("spawn")
        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # reenable robot collision
        for geom_id, _ in enumerate(self.sim.model.geom_bodyid):
            geom_name = self.sim.model.geom_id2name(geom_id)
            if geom_name is not None and "robot" in geom_name:
                contype, conaffinity = robot_col[geom_name]
                self.sim.model.geom_contype[geom_id] = contype
                self.sim.model.geom_conaffinity[geom_id] = conaffinity

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
        assert action.shape == (2,)
        action = np.concatenate([action, [0, 0]])
        super()._set_action(action)

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
            gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
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
            history["ac"].append(ac)
            obs, _, _, info = self.step(ac)
            history["obs"].append(obs)
            for k, v in info.items():
                history[k].append(v)
            if target_type == "gripper":
                gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
                d = target - gripper_xpos
            elif "object" in target_type:
                object_xpos = self.sim.data.get_site_xpos(target_type).copy()
                d = target - object_xpos
            step += 1

        # if np.linalg.norm(d) > threshold:
        #     print("move failed")
        # elif self._behavior == "push" and target_type == "object":
        #     goal_dist = np.linalg.norm(object_xpos - self.goal_pose["object"])
        #     print("goal object dist after push", goal_dist)

    def _get_joint_qpos(self, joint_name):
        return self.sim.data.get_joint_qpos(joint_name).copy()

    def _set_joint_qpos(self, joint_name, pose):
        """
        Takes in 7-Dof Pose eucl pos + quaternion [x,y,z, w, rx, ry, rz]
        """
        self.sim.data.set_joint_qpos(joint_name, pose)

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
        self._move(
            obj_target,
            history,
            target_type="object",
            speed=10,
            threshold=0.003,
        )

    def straight_push(self, history, object="object1", noise=0):
        # move gripper behind the block and oriented for a goal push
        block_xpos = self.sim.data.get_site_xpos(object).copy()
        spawn_xpos = self.sim.data.get_site_xpos("spawn").copy()
        goal_dir = (block_xpos - spawn_xpos) / np.linalg.norm(block_xpos - spawn_xpos)
        gripper_target = block_xpos - 0.05 * goal_dir
        self._move(gripper_target, history, speed=20, max_time=3)
        # push the block
        obj_target = block_xpos + 0.12 * goal_dir
        self._move(
            obj_target,
            history,
            target_type=object,
            speed=5,
            threshold=0.025,
            max_time=10,
            noise=noise,
        )

    def only_robot(self, history):
        # move gripper above cube
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip")
        gripper_target = gripper_xpos + [0, 0, 0.07]
        self._move(gripper_target, history)
        # move gripper to target robot pos
        gripper_xpos = self.sim.data.get_site_xpos("robot0:grip")
        gripper_target = self.goal_pose["gripper"]
        self._move(gripper_target, history, speed=10, threshold=0.025)

    def random_robot(self, history, ep_len):
        """Performs IID action sequence """
        for i in range(ep_len):
            ac = self.action_space.sample()
            history["ac"].append(ac)
            obs, _, _, info = self.step(ac)
            history["obs"].append(obs)
            for k, v in info.items():
                history[k].append(v)

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
        self._move(gripper_target, history, speed=100, max_time=3)
        past_acs = len(history["ac"])
        # generate temporally corellated noise
        u = np.zeros((ep_len, *self.action_space.shape))
        actions = np.zeros_like(u)
        actions[:past_acs] = history["ac"]
        for i in range(past_acs, ep_len):
            u[i] = self.action_space.sample()
            actions[i] = beta * u[i] + (1 - beta) * actions[i - 1]
        history["ac"] = actions

        for i in range(past_acs, ep_len):
            obs, _, _, info = self.step(actions[i])
            history["obs"].append(obs)
            for k, v in info.items():
                history[k].append(v)

    def random_robot_moving_object(self, history, ep_len, object="object1", noise=0):
        block_xpos = self.sim.data.get_site_xpos(object).copy()
        spawn_xpos = self.sim.data.get_site_xpos("spawn").copy()
        goal_dir = (block_xpos - spawn_xpos) / np.linalg.norm(block_xpos - spawn_xpos)
        moving_dist = 0.01 * goal_dir

        """Performs IID action sequence """
        for i in range(ep_len):
            ac = self.action_space.sample()
            history["ac"].append(ac)
            obs, _, _, info = self.step_auto_moving_obj(
                ac, moving_object=object, moving_dist=moving_dist
            )
            history["obs"].append(obs)
            for k, v in info.items():
                history[k].append(v)

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
        history["goal"] = self.goal.copy()
        ep_len = self._config.demo_length
        noise = self._config.action_noise
        beta = self._config.temporal_beta
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
        elif behavior == "temporal_random_robot":
            self.temporal_random_robot(history, ep_len, beta)
        elif behavior == "random_robot_moving_object":
            obj = np.random.choice(self._objects)
            history["pushed_obj"] = obj
            self.random_robot_moving_object(history, ep_len, object=obj)
        elif behavior == "straight_push":
            obj = np.random.choice(self._objects)
            history["pushed_obj"] = obj
            self.straight_push(history, object=obj, noise=noise)
        return history


if __name__ == "__main__":
    from src.config import argparser

    config, _ = argparser()
    env = ClutterPushEnv(config)
    env.reset()
    while True:
        env.render("human")
