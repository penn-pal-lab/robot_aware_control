import os
import h5py
import ipdb
import imageio
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces, utils
from mujoco_py.generated import const
from PIL import Image, ImageFilter
from skimage.filters import gaussian
from src.env.fetch.fetch_env import FetchEnv
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
            "object0:joint": [1.5, 0.75, 0.42, 1.0, 0.0, 0.0, 0.0],
            "object1:joint": [1.6, 0.75, 0.44, 1.0, 0.0, 0.0, 0.0],
            "object2:joint": [1.7, 0.75, 0.44, 1.0, 0.0, 0.0, 0.0],
        }
        self._objects = ["object0", "object1", "object2"]
        self._robot_pixel_weight = config.robot_pixel_weight
        reward_type = config.reward_type
        self._img_dim = config.img_dim
        self._camera_name = config.camera_name
        self._multiview = config.multiview
        self._camera_ids = config.camera_ids
        self._pixels_ob = config.pixels_ob
        self._norobot_pixels_ob = config.norobot_pixels_ob
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
            self.observation_space = spaces.Dict(
                dict(
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
            )

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
            obs = self.render("rgb_array", remove_robot=self._norobot_pixels_ob)
            robot = np.concatenate([grip_pos, grip_velp])
            return {
                "observation": obs.copy(),
                "achieved_goal": obs.copy(),
                "desired_goal": self.goal.copy(),
                "robot": robot.copy(),
            }
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

    def reset(self, spawn_info=None, goal_info=None):
        """
        spawn_info contains the spawn poses of the objects
        goal_info contains the goal poses, images, etc.
        """
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim(spawn_info)
        # if self.reward_type in ["inpaint", "inpaint-blur"] or self._norobot_pixels_ob:
        #     self._background_img = self._get_background_img()
        if goal_info is None:
            self.goal = self._sample_goal().copy()
        else:
            self.goal = self.set_goal(object, goal_info).copy()
        obs = self._get_obs()
        self._use_unblur = False
        return obs

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
        robot_noise = np.array([-1, 0, 0.5])
        gripper_target = robot_pos + robot_noise

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
        print("sampling objects")
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
        self, mode="rgb_array", camera_name=None, segmentation=False, remove_robot=False
    ):
        """
        If remove_robot, then use inpaint and mask to remove robot pixels
        remove_robot is used during dataset generation
        """
        if remove_robot:
            img = self.render()
            seg_mask = self.get_robot_mask()
            # update background img to most recent unoccluded pixels.
            # this is not useful for the current timestep, but will be useful
            # for future inpaintings
            # ipdb.set_trace()
            # self._background_img[~seg_mask] = img[~seg_mask].copy()
            # inpaint the img
            img[seg_mask] = self._background_img[seg_mask]
            return img

        if self._multiview:
            imgs = []
            for cam_id in self._camera_ids:
                camera_name = self.sim.model.camera_id2name(cam_id)
                img = super().render(
                    mode,
                    self._img_dim,
                    self._img_dim,
                    camera_name=camera_name,
                    segmentation=segmentation,
                )
                imgs.append(img)
            multiview_img = np.concatenate(imgs, axis=0)
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
        reset_mocap_welds(self.sim)
        self.sim.forward()
        # move robot gripper to above spawn
        gripper_target = self.sim.data.get_site_xpos("spawn") + np.array([0, 0, 0.5])

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

    def _get_joint_qpos(self, joint_name):
        return self.sim.data.get_joint_qpos(joint_name).copy()

    def _set_joint_qpos(self, joint_name, pose):
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
        size = "small"
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
                env = ClutterPushEnv(cfg)
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

            env = ClutterPushEnv(cfg)
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
    env = ClutterPushEnv(config)
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

    num_trajectories = 5000  # per worker
    num_workers = 20
    record = False
    behavior = "push"
    ep_len = 12  # gonna be off by -1 because of reset but whatever

    config, _ = argparser()
    config.invisible_demo = True
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
    env = ClutterPushEnv(config)
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


def collect_multiview_trajectories():
    """
    Collect multiview dataset with inpainting
    """
    from multiprocessing import Process

    num_trajectories = 5000  # per worker
    num_workers = 20
    record = False
    behavior = "random_robot"
    ep_len = 12  # gonna be off by -1 because of reset but whatever

    config, _ = argparser()
    config.large_block = True
    config.demo_dir = "demos/fetch_push_mv"
    config.multiview = True
    config.norobot_pixels_ob = True
    config.img_dim = 64
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
    env = ClutterPushEnv(config)
    for i in range(200):
        img = env.reset()
        img_path = os.path.join(config.demo_dir, f"{i}.png")
        imageio.imwrite(img_path, img)


def collect_object_demos():
    """Collect object only demonstrations"""
    config, _ = argparser()
    config.demo_dir = "demos/object_demos"
    config.multiview = True
    config.img_dim = 64
    config.push_dist = 0.25
    num_demos = 10
    record_gif = True

    os.makedirs(config.demo_dir, exist_ok=True)
    if record_gif:
        gif_dir = os.path.join(config.demo_dir, "gif")
        os.makedirs(gif_dir, exist_ok=True)
    env = ClutterPushEnv(config)
    stats_dict = defaultdict(list)
    for i in trange(num_demos, desc="Object Only Demos"):
        env.reset()
        push_path, imgs, info = env.make_push_object_demo()
        name = ""
        for obj in env._objects:
            if obj + "_straight_push" not in info:
                continue
            if info[obj + "_straight_push"]:
                name += "e"
            else:
                name += "h"

        stats_dict["len"].append(len(push_path))
        name = f"{name}_{i}"
        if record_gif:
            gif_path = os.path.join(gif_dir, f"{name}.gif")
            imageio.mimwrite(gif_path, imgs)
        path = os.path.join(config.demo_dir, f"{name}.hdf5")
        with h5py.File(path, "w") as hf:
            for k, v in info.items():
                hf.attrs[k] = v
            hf.create_dataset("frames", data=imgs, compression="gzip")
            hf.create_dataset("states", data=push_path, compression="gzip")

    # print out stats
    stats_path = os.path.join(config.demo_dir, "stats.txt")
    easy_len = stats_dict["len"]
    easy_stats = f"count: {len(easy_len)}, min len: {np.min(easy_len)}, max len: {np.max(easy_len)}, avg len: {np.mean(easy_len)}\n"
    with open(stats_path, "w") as f:
        f.writelines([easy_stats])
    print(easy_stats)


if __name__ == "__main__":
    from collections import defaultdict
    from copy import deepcopy

    import time
    import imageio
    from PIL import Image
    from src.config import argparser
    from torchvision.transforms import ToTensor
    from src.env.fetch.collision import CollisionSphere, CollisionBox
    from src.env.fetch.planar_rrt import PlanarRRT

    # plot_behaviors_per_cost()
    # plot_costs_per_behavior()
    # collect_trajectories()
    # collect_multiview_trajectories()
    # collect_cem_goals()
    # collect_object_demos()
    config, _ = argparser()
    env = ClutterPushEnv(config)
    env.reset()
    while True:
        img = env.render("rgb_array")
        imageio.imwrite("init.png", img)
        imageio.imwrite("goal.png", env._unblurred_goal)
        break
        env.reset()
    #     frames, imgs, info = env.make_push_object_demo()
    #     imageio.mimwrite(f"demo{i}.gif", imgs)
    # env.render("human")
    # time.sleep(1)
    # imageio.imwrite(f"goal{i}.png", goal)
    # imgs = []
    # for i in range(20):
    #     push = (1, 0, 0)
    #     env.step(push)
    #     env.reset()
    #     img = env.render("rgb_array")
    #     imgs.append(img)
    #     goal = env.goal
    #     imageio.imwrite("goal.png", goal)
    #     break
    #     img = env.render("rgb_array")
    #     imageio.imwrite("test.png", img)
    #     break
    #     tensor = ToTensor()(img)
    #     print(tensor.shape)

    # img_width = 256
    # s = 5
    # w = 2 * img_width
    # t = (((w - 1) / 2) - 0.5) / s
    # goal = np.array(Image.open("code.png"))
    # goal = np.uint8(255 * gaussian(goal, sigma=s, truncate=t, mode="nearest", multichannel=True))
    # imageio.imwrite("blurredcode.png", goal)
