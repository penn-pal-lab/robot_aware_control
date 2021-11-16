import copy
from functools import partial
import os
from collections import defaultdict

import numpy as np
from gym import spaces
from src.env.robotics.masks.base_mask_env import MaskEnv
from src.env.robotics.utils import (ctrl_set_action, mocap_set_action,
                                    reset_mocap2body_xpos, reset_mocap_welds)

DEBUG = False

class LocobotOcclusionEnv(MaskEnv):
    def __init__(self, config):
        self._config = config
        modified =  config.modified
        model_path = f"locobot_occlusion.xml"
        model_path = os.path.join("locobot", model_path)

        initial_qpos = None
        n_actions = 4
        n_substeps = 20
        seed = config.seed
        np.random.seed(seed)
        self._img_width = 800
        self._img_height = 800
        self._render_device = config.render_device
        if modified:
            self._joints = [f"joint_{i}" for i in range(1, 8)]
            self._gripper_joints = ['robot0:r_gripper_finger_joint', 'robot0:l_gripper_finger_joint']
        else:
            self._joints = [f"joint_{i}" for i in range(1, 8)]

        self._geoms = {
            "robot-geom-0",
            "robot-geom-1",
            "robot-geom-2",
            "robot-geom-3",
            "robot-geom-4",
            "robot-geom-5",
            "robot-geom-6",
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

        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)

        self._camera_name = "main_cam"
        # self._joints.append("gripper_revolute_joint")
        self._joint_references = [
            self.sim.model.get_joint_qpos_addr(x) for x in self._joints
        ]
        self._joint_vel_references = [
            self.sim.model.get_joint_qvel_addr(x) for x in self._joints
        ]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")

        self._objects = ["object1"]

        # workspace boundaries for eef
        # self._ws_low = [0.24, -0.17, 0.05]
        self._ws_low = [0.24, -0.17, 0.05]
        self._ws_high = [0.33, 0.17, 0.3]

        # modify camera R
        self.initial_sim_state = None


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
        mask = np.zeros(mask_dim, dtype=bool)
        # TODO: change these to include the robot base
        # ignore_parts = {"finger_r_geom", "finger_l_geom"}
        ignore_parts = {}
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if name is not None:
                if name in ignore_parts:
                    continue
                if name in self._geoms or "robot0" in name:
                    mask[ids == i] = True
        return mask

    def generate_masks(self, qpos_data, width=None, height=None):
        joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]
        masks = []
        for qpos in qpos_data:
            self.sim.data.qpos[joint_references] = qpos
            self.sim.forward()
            mask = self.get_robot_mask(width, height)
            masks.append(mask)
        masks = np.asarray(masks, dtype=np.bool)
        return masks

    def reset(self, initial_state=None, init_robot_qpos=True):
        """Reset the robot and block pose

        Args:
            initial_state ([type], optional): dictionary containing the robot / block poses. Defaults to None.
            init_robot_qpos (bool, optional): initialize qpos from initial_state if true. else use eef pos.

        Returns:
            [type]: [description]
        """
        if self.initial_sim_state is None:
            if self._config.modified:
                self.sim.data.qpos[self._joint_references] = [-0.25862757, -1.20163741,  0.32891832,  1.42506277, -0.10650079,  1.43468923, 0.06129823]
            else:
                # first move the arm above to avoid object collision
                # robot_above_qpos = [0.0, 0.1, 0.2393125, 0.63018035, 0.0, 0, 0]
                robot_above_qpos = [0.8, 0.1, 0.2393125, 0.63018035, 0.0, 0, 0]
                self.sim.data.qpos[self._joint_references] = robot_above_qpos
                self.sim.forward()
            self.initial_sim_state = copy.deepcopy(self.sim.get_state())
        else:
            self.sim.set_state(self.initial_sim_state)
        reset_mocap_welds(self.sim)
        reset_mocap2body_xpos(self.sim)
        # then sample object initialization
        self._sample_objects()

        # eef_target_pos = [0.3, 0.0, 0.15]
        # # some noise to the x/y of the eef initial pos
        # noise = np.random.uniform(-0.03, 0.03, size = 2)
        # eef_target_pos[:2] += noise
        # self._move(eef_target_pos, threshold=0.01, max_time=100, speed=10)

        if initial_state is not None:
            if init_robot_qpos:
                self.sim.data.qpos[self._joint_references] = initial_state["qpos"].copy()
            else:
                self._move(initial_state["states"][:3], threshold=0.01, max_time=100, speed=10)
            self.sim.data.set_joint_qpos("object1:joint", initial_state["obj_qpos"].copy())
            self.sim.forward()
        return self._get_obs()

    def step(self, action, clip=True):
        action = np.asarray(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if clip:
            # check if applying action will violate the workspace boundary, if so, clip it.
            curr_eef_state = self.get_gripper_world_pos()
            next_eef_state = curr_eef_state + (action[:3] * 0.05)

            next_eef_state = np.clip(next_eef_state, self._ws_low, self._ws_high)
            clipped_ac = (next_eef_state - curr_eef_state) / 0.05
            action[:3] = clipped_ac
        self._set_action(action)
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
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

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
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        # print(gripper_ctrl)
        # assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, [gripper_ctrl, gripper_ctrl]])
        # Apply action to simulation.
        ctrl_set_action(self.sim, action)
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
        if DEBUG:
            img = np.zeros((48,64,3))
            masks = np.zeros((48,64,1))
        else:
            img = self.render("rgb_array")
            masks = self.get_robot_mask()
        gripper_xpos = self.get_gripper_world_pos()
        # assume 0 for rotation, gripper force
        states = np.array([*gripper_xpos, 0, 0])
        qpos = self.sim.data.qpos[self._joint_references].copy()
        # object qpos
        obj_qpos = self.sim.data.get_joint_qpos("object1:joint").copy()
        return {"observation": img, "masks": masks, "states": states, "qpos": qpos, "obj_qpos": obj_qpos}

    def render(self, mode="rgb_array", camera_name=None, segmentation=False, width=None, height=None, depth=False):
        if width is None or height is None:
            width, height = self._img_width, self._img_height
        if camera_name is None:
            camera_name = "main_cam"
        if mode == "rgb_array":
            data = self.sim.render(
                width,
                height,
                camera_name=camera_name,
                segmentation=segmentation,
                device_id=self._render_device,
                depth=depth
            )
            if depth:
                img, _depth = data
                # extent = self.sim.model.stat.extent
                # near = self.sim.model.vis.map.znear * extent
                # far = self.sim.model.vis.map.zfar * extent
                # _depth = near / (1 - _depth * (1 - near / far))

                return img[::-1], _depth[::-1]
            else:
                # original image is upside-down, so flip it
                return data[::-1]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _sample_objects(self):
        # set objects in radius around spawn
        center = self.sim.data.get_site_xpos("blockspawn")[:2]
        spawn_id = self.sim.model.site_name2id("blockspawn")
        radius = self.sim.model.site_size[spawn_id][0]
        failed = False
        sampled_points = []
        for obj in self._objects:
            # reject sample if it overlaps with previous objects
            # reject sample if it's too close to the spawn point where the robot is
            for i in range(1000):
                no_overlap = True
                xy = self._sample_from_circle(center, radius)
                # if np.linalg.norm(xy - center) < 0.08:
                #     continue

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
        history=None,
        target_type="gripper",
        max_time=100,
        threshold=0.01,
        speed=10,
        noise=0,
        gripper=0.05,
        clip=True
    ):
        if target_type == "gripper":
            gripper_xpos = self.get_gripper_world_pos()
            d = target - gripper_xpos
        elif "object" in target_type:
            object_xpos = self.sim.data.get_site_xpos(target_type).copy()
            d = target - object_xpos
        step = 0
        while np.linalg.norm(d) > threshold and step < max_time:
            # add some random noise to ac
            if noise > 0:
                d[:3] = d[:3] + np.random.uniform(-noise, noise, size=2)
            ac = np.clip(d[:3] * speed, -1, 1)
            pad_ac = [*ac, gripper]
            if history is not None:
                history["ac"].append(pad_ac)

            obs, _, _, info = self.step(pad_ac, clip=clip)
            if history is not None:
                history["obs"].append(obs)
                for k, v in info.items():
                    history[k].append(v)
            if target_type == "gripper":
                gripper_xpos = self.get_gripper_world_pos()
                d = target - gripper_xpos
            elif "object" in target_type:
                object_xpos = self.sim.data.get_site_xpos(target_type).copy()
                d = target - object_xpos
            step += 1

    def generate_demo(self, noise_level):
        """
        Runs a hard coded behavior and stores the episode
        Returns a dictionary with observation, action
        """
        # initialize place pos
        place_xpos = self.place_xpos = np.array([0.3, 0.11, 0.17])
        place_noise = np.random.uniform([-0.03, -0.02], [0.03, 0.03], size=2)
        place_xpos[:2] += place_noise
        body_idx = self.sim.model.body_name2id("placebody")
        self.sim.model.body_pos[body_idx] = place_xpos.copy()
        # initialize the place  marker
        obs = self.reset()
        if DEBUG:
            self.render("human")
        history = defaultdict(list)
        history["obs"].append(obs)
        self.pick_place(place_xpos, history, noise_level)
        return history


    def pick_place(self, place_xpos, history, noise_level="none", max_actions=14):
        """first move robot gripper over random object,
        then grasp
        """
        total_steps = 0
        max_actions = 14

        obj = np.random.choice(self._objects)
        history["pushed_obj"] = obj

        # move gripper behind the block and oriented for a goal push
        block_xpos = self.sim.data.get_site_xpos(obj).copy()
        above_block_xpos = block_xpos.copy()
        above_block_xpos[2] += 0.05
        # move robot above slightly above block
        target = above_block_xpos
        if noise_level == "high":
            noise = 0.05
            z_noise = 0.04
            gripper_noise = 0.005
        elif noise_level == "med":
            noise = 0.03
            z_noise = 0.02
            gripper_noise = 0.005
        elif noise_level == "none":
            noise = 0.00
            z_noise = 0.00
            gripper_noise = 0.00
        speed = 40
        gripper_xpos = self.get_gripper_world_pos()
        d = target - gripper_xpos
        step = 0
        while np.linalg.norm(d[:2]) > 0.01 and step < 3:
            # add some random noise to ac
            if noise > 0:
                d[:2] = d[:2] + np.random.uniform(-noise, noise, size=2)
                d[2] = d[2] + np.random.uniform(-z_noise, z_noise)
            ac = np.clip(d[:3] * speed, -1, 1)
            gripper_ac = -0.002  + np.random.uniform(-gripper_noise, gripper_noise) # start closing
            pad_ac = [*ac, gripper_ac]
            pad_ac = np.clip(pad_ac, self.action_space.low, self.action_space.high)
            if history is not None:
                history["ac"].append(pad_ac)

            obs, _, _, info = self.step(pad_ac)
            if DEBUG:
                self.render("human")
            if history is not None:
                history["obs"].append(obs)
                for k, v in info.items():
                    history[k].append(v)
            gripper_xpos = self.get_gripper_world_pos()
            d = target - gripper_xpos
            step += 1
        total_steps += step
        # print("move above", step)
        # descend onto block, and close gripper
        block_xpos = self.sim.data.get_site_xpos(obj).copy()
        block_xpos[2] -= 0.01
        target = block_xpos
        if noise_level == "high":
            noise = 0.02
        elif noise_level == "med":
            noise = 0.01
        elif noise_level == "none":
            noise = 0.0
        speed = 40
        gripper_xpos = self.get_gripper_world_pos()
        d = target - gripper_xpos
        step = 0
        while np.linalg.norm(d) > 0.01 and step < 3:
            # add some random noise to ac
            if noise > 0:
                d[:2] = d[:2] + np.random.uniform(-noise, noise, size=2)
                d[2] = d[2] + np.random.uniform(-z_noise, z_noise)
            ac = np.clip(d[:3] * speed, -1, 1)
            gripper_ac = -0.01 + np.random.uniform(-gripper_noise, gripper_noise)# close
            pad_ac = [*ac, gripper_ac]
            pad_ac = np.clip(pad_ac, self.action_space.low, self.action_space.high)
            if history is not None:
                history["ac"].append(pad_ac)

            obs, _, _, info = self.step(pad_ac)
            if DEBUG:
                self.render("human")
            if history is not None:
                history["obs"].append(obs)
                for k, v in info.items():
                    history[k].append(v)
            gripper_xpos = self.get_gripper_world_pos()
            d = target - gripper_xpos
            step += 1

        total_steps += step
        # print("pick", step)

        # Place primitive
        if noise_level == "high":
            noise = 0.04
            gripper_noise = 0.02
        elif noise_level == "med":
            noise = 0.02
            gripper_noise = 0.005
        elif noise_level == "none":
            noise = 0.0
            gripper_noise = 0.0
        speed = 40
        gripper_xpos = self.get_gripper_world_pos()
        block_xpos = self.sim.data.get_site_xpos(obj).copy()

        target = block_xpos.copy()
        target[2] = 0.2
        d = target - gripper_xpos
        step = 0
        # first lift it up
        while np.linalg.norm(d) > 0.01 and step < 4:
            # add some random noise to ac
            if noise > 0:
                d[:2] = d[:2] + np.random.uniform(-noise, noise, size=2)
                d[2] = d[2] + np.random.uniform(-z_noise, z_noise)
            ac = np.clip(d[:3] * speed, -1, 1)
            gripper_ac = -0.01 + np.random.uniform(-gripper_noise, gripper_noise)# close
            pad_ac = [*ac, gripper_ac]
            pad_ac = np.clip(pad_ac, self.action_space.low, self.action_space.high)
            if history is not None:
                history["ac"].append(pad_ac)

            obs, _, _, info = self.step(pad_ac)
            if DEBUG:
                self.render("human")
            if history is not None:
                history["obs"].append(obs)
                for k, v in info.items():
                    history[k].append(v)
            gripper_xpos = self.get_gripper_world_pos()
            d = target - gripper_xpos
            step += 1
        # print("lift", step)
        total_steps += step

        # move it to a side.
        if noise_level == "high":
            noise = 0.03
        elif noise_level == "med":
            noise = 0.015
        elif noise_level == "none":
            noise = 0.0
        speed = 40
        gripper_xpos = self.get_gripper_world_pos()

        # move it on top of the platform
        target = place_xpos
        d = target - gripper_xpos
        step = 0
        while total_steps < max_actions:

            if np.linalg.norm(d) > 0.01:
                # add some random noise to ac
                if noise > 0:
                    d[:2] = d[:2] + np.random.uniform(-noise, noise, size=2)
                    d[2] = d[2] + np.random.uniform(-z_noise, z_noise)
                ac = np.clip(d[:3] * speed, -1, 1)
                gripper_ac = -0.01 + np.random.uniform(-gripper_noise, gripper_noise)# close
                pad_ac = [*ac, gripper_ac]
                pad_ac = np.clip(pad_ac, self.action_space.low, self.action_space.high)
            else:
                pad_ac = np.zeros(4)

            if history is not None:
                history["ac"].append(pad_ac)

            obs, _, _, info = self.step(pad_ac)
            if DEBUG:
                self.render("human")
            if history is not None:
                history["obs"].append(obs)
                for k, v in info.items():
                    history[k].append(v)
            gripper_xpos = self.get_gripper_world_pos()
            d = target - gripper_xpos
            total_steps += 1

        block_xpos = self.sim.data.get_site_xpos(obj).copy()
        # print(block_xpos[2])
        success =  np.linalg.norm(block_xpos - place_xpos) < 0.02
        history["success"] = success
        # print("total", len(history["ac"]), total_steps)
        # print("success", success)

    def get_gripper_world_pos(self):
        return self.sim.data.get_site_xpos("robot0:grip").copy()

    def get_gripper_val(self):
        if self._config.modified:
            gripper_joints = self._gripper_joints
        else:
            gripper_joints = self._joints[-2:]

        return np.array([self.sim.data.get_joint_qpos(g).copy() for g in gripper_joints])

    def set_gripper_val(self, values):
        if self._config.modified:
            gripper_joints = self._gripper_joints
        else:
            gripper_joints = self._joints[-2:]
        # assumes right, then left gripper value
        self.sim.data.set_joint_qpos(gripper_joints[0], values[0])
        self.sim.data.set_joint_qpos(gripper_joints[1], values[1])
        self.sim.forward()

def compute_iou(prediction, target):
    I = prediction & target
    U = prediction | target
    IoU = I.sum() / U.sum()
    return IoU

def generate_iou_data():
    from src.config import argparser
    from src.utils.mujoco import init_mjrender_device

    config, _ = argparser()
    init_mjrender_device(config)
    config.gpu = 0
    config.modified = False

    threshold = 0.01
    FOLDER_PATH = "depth_exp"
    os.makedirs(FOLDER_PATH, exist_ok=True)
    def imwrite(path, img):
        # crop the image's height by half
        r,c = img.shape[:2]
        img = img[:r//2, c//4:3*c//4 ]
        imageio.imwrite(os.path.join(FOLDER_PATH, path), img)

    DEBUG = False
    x_distances = [0.18, 0.181, 0.182, 0.183, 0.184, 0.185, 0.186, 0.187, 0.188, 0.189, 0.19, 0.195, 0.2, 0.21, 0.22, 0.23, 0.25]
    x_ious = []
    for i, x_distance in enumerate(x_distances):
        env = LocobotOcclusionEnv(config)
        obs = env.reset()
        # set the object somewhere
        joint = "object1:joint"
        # flush pose
        # obj_pose = [0.18,-0.012, 0.09, 1,0,0,0]
        # normal pose
        obj_pose = [x_distance, -0.012, 0.1, 1,0,0,0]
        env.sim.data.set_joint_qpos(joint, obj_pose)
        env.sim.forward()
        img, depth = env.render("rgb_array", depth=True)

        extent = env.sim.model.stat.extent
        img = np.uint8(img)
        depth_img = np.uint8(255 * depth)
        imwrite(f"{x_distance}_img_0.png", img)
        imwrite(f"{x_distance}_depth_0.png",depth_img)
        occ_img = img
        occ_depth = depth
        true_occ_mask =  env.get_robot_mask()
        true_occ_mask_img = np.uint8(255 * true_occ_mask)
        imwrite(f"{x_distance}_true_occ_robot_mask.png", true_occ_mask_img)

        # now set the object out of frame
        pose = [10, 0, 0, 1, 0, 0, 0]
        env.sim.data.set_joint_qpos("object1:joint", pose)
        env.sim.forward()

        img, depth = env.render("rgb_array", depth=True)
        img = np.uint8(img)
        depth_img = np.uint8(255 * depth)
        imwrite(f"{x_distance}_img_1.png", img)
        imwrite(f"{x_distance}_depth_1.png",depth_img)
        non_occ_img = img
        non_occ_depth = depth
        non_occ_mask = env.get_robot_mask()
        non_occ_mask_img = np.uint8(255 * non_occ_mask)
        imwrite(f"{x_distance}_no_occ_robot_mask.png", non_occ_mask_img)

        # compute areas where objects occlude the robot.
        depth_diff = np.abs(occ_depth - non_occ_depth) > threshold
        depth_diff_img = np.uint8(depth_diff) * 255
        imwrite(f"{x_distance}_mask_diff.png", depth_diff_img)
        non_occ_mask_img[depth_diff] = 0
        imwrite(f"{x_distance}_computed_occ_robot_mask.png", non_occ_mask_img)

        non_occ_mask[depth_diff] = 0
        iou = compute_iou(non_occ_mask, true_occ_mask)
        x_ious.append(iou)
        print(f"{x_distance} IOU:", iou)

    all_xdist_iou = [f"{x}: {iou}" for x, iou in zip(x_distances, x_ious)]
    with open(os.path.join(FOLDER_PATH, "stats.txt"), "w") as f:
        f.writelines(all_xdist_iou)

    with open(os.path.join(FOLDER_PATH, "data.pkl"), "wb") as f:
        data = {"x_distances": x_distances, "x_ious": x_ious}
        pickle.dump(data, f)

def plot_iou_data():
    import matplotlib.pyplot as plt

    FOLDER_PATH = "depth_exp/"
    with open(os.path.join(FOLDER_PATH, "data.pkl"), "rb") as f:
        data = pickle.load(f)
        x = np.array(data["x_distances"]) * 100 # convert to cm
        x = x - x[0] # relative dist to robot
        y = data["x_ious"]
    plt.plot(x,y, marker="x")
    plt.xlabel("Object distance to robot (cm)")
    plt.ylabel("IoU of robot mask")
    plt.title("Quality of computed mask over distance to robot")
    plt.axvline(x=1, label="Depth Threshold", linestyle="dashed", color="red")

    plt.savefig(os.path.join(FOLDER_PATH,"iou.png"), dpi=400)
    plt.savefig(os.path.join(FOLDER_PATH,"iou.pdf"))


if __name__ == "__main__":
    import sys
    import imageio
    import pickle

    # generate_iou_data()
    plot_iou_data()