#! /usr/bin/env python

from __future__ import print_function
import numpy as np
import os
import random
import cv2
from cv_bridge import CvBridge
import imageio
import pathlib
import h5py
from pupil_apriltags import Detector
from time import gmtime, strftime
from tqdm import trange
import ipdb

# ROS
import rospy
import actionlib
from sensor_msgs.msg import Image

# Learning
import torch
import torchvision.transforms as tf

# Defined by us
from eef_control.msg import *
from locobot_rospkg.nodes.data_collection_client import (
    eef_control_client,
    gaussian_push,
    PUSH_HEIGHT,
    DEFAULT_PITCH,
    DEFAULT_ROLL,
)

from src.config import argparser
from src.prediction.models.dynamics import SVGConvModel
from src.prediction.losses import ImgL2Cost
from src.dataset.locobot.locobot_model import LocobotAnalyticalModel
from src.cem.cem import CEMPolicy
from src.dataset.robonet.robonet_dataset import normalize
from src.utils.state import DemoGoalState, State


class Visual_MPC(object):
    def __init__(self, config, device="cuda"):
        # Creates the SimpleActionClient, passing the type of the action
        self.control_client = actionlib.SimpleActionClient(
            "eef_control", eef_control.msg.PoseControlAction
        )

        self.img_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.img_callback
        )
        self.depth_sub = rospy.Subscriber(
            "/camera/depth/image_rect_raw", Image, self.depth_callback
        )
        self.cv_bridge = CvBridge()
        self.img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth = np.zeros((480, 640), dtype=np.uint16)

        self.device = device
        self.model = None
        self.config = config

        w, h = config.image_width, config.image_height
        self._img_transform = tf.Compose([tf.ToTensor(), tf.Resize((h, w))])
        self.t = 1
        self.target_img = None

        model = SVGConvModel(config)
        ckpt = torch.load(config.dynamics_model_ckpt, map_location=config.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        self.policy = CEMPolicy(
            config,
            model,
            init_std=config.cem_init_std,
            action_candidates=config.action_candidates,
        )

    def img_callback(self, data):
        self.img = self.cv_bridge.imgmsg_to_cv2(data)

    def depth_callback(self, data):
        self.depth = self.cv_bridge.imgmsg_to_cv2(data)

    def get_camera_pose_from_apriltag(self, detector=None):
        print("[INFO] detecting AprilTags...")
        if detector is None:
            detector = Detector(
                families="tag36h11",
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        results = []
        results = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[612.45, 612.45, 330.55, 248.61],
            tag_size=0.0353,
        )
        print("[INFO] {} total AprilTags detected".format(len(results)))

        if len(results) == 0:
            return None, None
        elif len(results) > 1:
            print("[Error] More than 1 AprilTag detected!")

        # loop over the AprilTag detection results
        for r in results:
            pose_t = r.pose_t
            pose_R = r.pose_R
        # Tag pose w.r.t. camera
        return pose_t, pose_R

    def read_target_image(self):
        if self.config.test_without_robot:
            with h5py.File(self.config.h5py_path, "r") as hf:
                IMAGE_KEY = "observations"
                images = hf[IMAGE_KEY][:]
                images = torch.stack([self._img_transform(i) for i in images]).to(
                    self.device
                )
                self.target_img = images[self.t + 1]

                gt_actions = hf["actions"][self.t].astype(np.float32)
                print("gt action:", gt_actions)
        else:
            if self.target_img is None:
                print("Collect target image before MPC first!")
                return
            self.target_img = self._img_transform(self.target_img).to(self.device)

    def collect_target_img(self, eef_target):
        """ set up the scene and collect goal image """
        control_result = eef_control_client(
            self.control_client,
            target_pose=[*eef_target, PUSH_HEIGHT, DEFAULT_PITCH, DEFAULT_ROLL],
        )
        self.target_img = np.copy(self.img)

    def go_to_start_pose(self, eef_start):
        """ set up the starting scene """
        control_result = eef_control_client(
            self.control_client,
            target_pose=[*eef_start, PUSH_HEIGHT, DEFAULT_PITCH, DEFAULT_ROLL],
        )
        input("Move the object close to the EEF. Press Enter to continue...")
        self.start_img = np.copy(self.img)
        self.start_img = self._img_transform(self.start_img).to(self.device)
        self.start_img = self.start_img.cpu().clamp_(0, 1).numpy()
        self.start_img = np.transpose(self.start_img, axes=(1, 2, 0))
        self.start_img = np.uint8(self.start_img * 255)
        # self.start_img = cv2.cvtColor(self.start_img, cv2.COLOR_BGR2RGB)

    # @torch.no_grad()
    # def vanilla_rollout(self, a_in):
    #     """
    #     Rollout using Vanilla Locobot Model
    #     a_in: numpy array (5, ), input action
    #     """
    #     if self.model is None:
    #         print("No vanilla model loaded")
    #         return

    #     image, mask, robot, heatmap, action, next_image, next_mask, next_robot, next_heatmap \
    #         = None, None, None, None, None, None, None, None, None
    #     action = torch.from_numpy(a_in).type(torch.float32).to(self.device)
    #     action = torch.unsqueeze(action, 0)

    #     if self.config.test_without_robot:
    #         with h5py.File(self.config.h5py_path, "r") as hf:
    #             IMAGE_KEY = "observations"
    #             MASK_KEY = "masks"

    #             images = hf[IMAGE_KEY][:]
    #             states = hf["states"][:].astype(np.float32)
    #             if states.shape[-1] != self.config.robot_dim:
    #                 assert self.config.robot_dim > states.shape[-1]
    #                 pad = self.config.robot_dim - states.shape[-1]
    #                 states = np.pad(states, [(0, 0), (0, pad)])

    #             low = np.array([0.015, -0.3, 0.1, 0, 0], dtype=np.float32)
    #             high = np.array([0.55, 0.3, 0.4, 1, 1], dtype=np.float32)
    #             # normalize the locobot xyz to the 0-1 bounds
    #             # rotation, gripper are always zero so it doesn't matter
    #             states = normalize(states, low, high)
    #             robot = "locobot"
    #             gt_actions = hf["actions"][:].astype(np.float32)

    #             # masks = hf[MASK_KEY][:].astype(np.float32)

    #             qpos = hf["qpos"][:].astype(np.float32)
    #             if qpos.shape[-1] != self.config.robot_joint_dim:
    #                 assert self.config.robot_joint_dim > qpos.shape[-1]
    #                 pad = self.config.robot_joint_dim - qpos.shape[-1]
    #                 qpos = np.pad(qpos, [(0, 0), (0, pad)])

    #             # preprocessing
    #             images = torch.stack([self._img_transform(i)
    #                                  for i in images]).to(self.device)
    #             gt_actions = torch.from_numpy(gt_actions).to(self.device)
    #             states = torch.from_numpy(states).to(self.device)

    #             image = torch.unsqueeze(images[self.t], 0)
    #             gt_action = torch.unsqueeze(gt_actions[self.t], 0)
    #     else:
    #         """ Real robot visual MPC """
    #         image = torch.unsqueeze(self._img_transform(np.copy(self.img)),
    #                                 0).to(self.device)

    #     b = min(image.shape[0], 10)
    #     self.model.init_hidden(b)
    #     x_pred, curr_skip, _, _, _, _ \
    #         = self.model.forward(image, mask, robot, heatmap, action,
    #                              next_image, next_mask, next_robot, next_heatmap)
    #     x_pred, x_pred_mask = x_pred[:, :3], x_pred[:, 3].unsqueeze(1)
    #     x_pred = (1 - x_pred_mask) * image + (x_pred_mask) * x_pred

    #     img_pred = x_pred.squeeze().cpu().clamp_(0, 1).numpy()
    #     img_pred = np.transpose(img_pred, axes=(1, 2, 0))
    #     img_pred = np.uint8(img_pred * 255)
    #     # img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
    #     return img_pred, x_pred.squeeze()

    # @torch.no_grad()
    # def rollout(self):
    #     pass

    # def cem(self):
    #     self.read_target_image()
    #     goal_visual = self.target_img.cpu().clamp_(0, 1).numpy()
    #     goal_visual = np.transpose(goal_visual, axes=(1, 2, 0))
    #     goal_visual = np.uint8(goal_visual * 255)
    #     goal_visual = cv2.cvtColor(goal_visual, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite("figures/goal_visual.png", goal_visual)

    #     goal_state = State(img=self.target_img)
    #     rew_func = ImgL2Cost(self.config)
    #     all_rews = []

    #     action_samples = gaussian_push(nactions=100)
    #     action_samples[:, 2:] = 0
    #     for i in range(action_samples.shape[0]):
    #         img_pred, x_pred = self.vanilla_rollout(action_samples[i])
    #         # cv2.imwrite("figures/img_pred_" + str(i) + ".png", img_pred)
    #         imageio.mimwrite("figures/img_pred_" + str(i) +
    #                          ".gif", [self.start_img, img_pred], fps=2)
    #         pred_state = State(img=x_pred)
    #         rew = rew_func(pred_state, goal_state)
    #         all_rews.append(rew)
    #     all_rews = np.array(all_rews)
    #     best_action_id = np.argmax(all_rews)
    #     best_action = action_samples[best_action_id]
    #     print("best action idx:", best_action_id)
    #     print("best action:", best_action)
    #     return best_action

    def cem(self):
        self.read_target_image()
        goal_visual = self.target_img.cpu().clamp_(0, 1).numpy()
        goal_visual = np.transpose(goal_visual, axes=(1, 2, 0))
        self.goal_visual = goal_visual = np.uint8(goal_visual * 255)

        start_visual = self.start_img
        imageio.imwrite(
            "figures/start_goal.png", np.concatenate([start_visual, goal_visual], 1)
        )
        start = State(img=start_visual)
        goal = DemoGoalState(imgs=[goal_visual])
        actions = self.policy.get_action(start, goal, 0, 0)
        return actions

    def execute_action(self, action):
        control_result = eef_control_client(self.control_client, target_pose=[])
        end_xy = [
            control_result.end_pose[0] + action[0],
            control_result.end_pose[1] + action[1],
        ]
        control_result = eef_control_client(
            self.control_client,
            target_pose=[*end_xy, PUSH_HEIGHT, DEFAULT_PITCH, DEFAULT_ROLL],
        )

    def execute_open_loop(self, actions):
        img = np.copy(self.img)
        img = self._img_transform(img)
        img = img.clamp_(0, 1).numpy()
        img = np.transpose(img, axes=(1, 2, 0))
        img = np.uint8(img * 255)

        img_goal = np.concatenate([img, self.goal_visual], 1)
        gif = [img_goal]
        for ac in actions:  # execute open loop actions for now
            vmpc.execute_action(ac)
            img = np.copy(self.img)
            img = self._img_transform(img)
            img = img.clamp_(0, 1).numpy()
            img = np.transpose(img, axes=(1, 2, 0))
            img = np.uint8(img * 255)
            img_goal = np.concatenate([img, self.goal_visual], 1)
            gif.append(img_goal)
        imageio.mimwrite("open_loop.gif", gif, fps=2)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf, _ = argparser()
    cf.device = device
    cf.test_without_robot = False
    # cf.h5py_path = "/home/pallab/locobot_ws/src/eef_control/data/data_2021-04-14_18:54:25.hdf5"
    CKPT_PATH = (
        "/home/pallab/locobot_ws/src/roboaware/checkpoints/locobot_689_ckpt_213000.pt"
    )
    # cf.h5py_path = "/mnt/ssd1/pallab/locobot_data/data_2021-03-20/data_2021-03-20_19_05_02.hdf5"
    # CKPT_PATH = "/mnt/ssd1/pallab/pal_ws/src/roboaware/checkpoints/locobot_689_ckpt_213000.pt"
    cf.dynamics_model_ckpt = "locobot_689_tile_ckpt_136500.pt"
    cf.debug_cem = True
    cf.cem_init_std = 0.015
    cf.action_candidates = 300

    # Initializes a rospy node so that the SimpleActionClient can
    # publish and subscribe over ROS.
    rospy.init_node("visual_mpc_client")

    vmpc = Visual_MPC(config=cf)
    vmpc.get_camera_pose_from_apriltag()

    eef_target_pos = [0.33, 0]
    vmpc.collect_target_img(eef_target_pos)
    vmpc.go_to_start_pose(eef_start=[eef_target_pos[0], eef_target_pos[1] - 0.1])
    actions = vmpc.cem()  # returns action trajectory
    print(actions)
    input("execute actions?")
    vmpc.execute_open_loop(actions)
