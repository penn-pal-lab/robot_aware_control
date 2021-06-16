#! /usr/bin/env python

from __future__ import print_function
from locobot_rospkg.nodes.data_collection_client import PUSH_HEIGHT
from src.env.robotics.masks.franka_mask_env import FrankaMaskEnv

import sys
import time
from time import gmtime, strftime
import os
import pickle

from typing import Tuple

import actionlib
import cv2
import imageio
import ipdb
import numpy as np
import rospy
import torch
import torchvision.transforms as tf
from cv_bridge import CvBridge

from eef_control.msg import *
from locobot_rospkg.nodes.franka_IK_client import FrankaIKClient
from locobot_rospkg.nodes.franka_control_client import FrankaControlClient

from scipy.spatial.transform.rotation import Rotation
from sensor_msgs.msg import Image
from src.cem.cem import CEMPolicy
from src.config import create_parser, str2bool

from src.prediction.models.dynamics import SVGConvModel
from src.utils.camera_calibration import camera_to_world_dict, LOCO_FRANKA_DIFF
from src.utils.state import DemoGoalState, State

start_offset = 0.15

# locobot frame
START_POS = {
    "left": np.array([0.655, -0.079]),
    "right": np.array([0.625, 0.191]),
    "forward": np.array([0.565 + 0.02, 0.061]),
}

CAMERA_CALIB = camera_to_world_dict["franka_c0"]
PUSH_HEIGHT = 0.15


class Visual_MPC(object):
    def __init__(self, config, device="cuda"):
        # Creates the SimpleActionClient, passing the type of the action
        self.control_client = FrankaControlClient()
        self.img_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.img_callback
        )
        self.cv_bridge = CvBridge()
        self.img = np.zeros((480, 640, 3), dtype=np.uint8)

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

        self.ik_solver = FrankaIKClient()
        self.env_thick = FrankaMaskEnv()

        camTbase = CAMERA_CALIB
        self.set_camera_calibration(camTbase)

        self.policy = CEMPolicy(
            config,
            model,
            init_std=config.cem_init_std,
            action_candidates=config.action_candidates,
            horizon=config.horizon,
            opt_iter=config.opt_iter,
            cam_ext=camTbase,
            franka_ik=self.ik_solver
        )

    def img_callback(self, data):
        self.img = self.cv_bridge.imgmsg_to_cv2(data)


    def set_camera_calibration(self, camTbase):
        rot_matrix = camTbase[:3, :3]
        cam_pos = camTbase[:3, 3]
        rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
        cam_rot = Rotation.from_matrix(rot_matrix) * rel_rot

        cam_id = 0
        offset = [0, 0, 0]
        print("applying offset", offset)
        self.env_thick.sim.model.cam_pos[cam_id] = cam_pos + offset
        cam_quat = cam_rot.as_quat()
        self.env_thick.sim.model.cam_quat[cam_id] = [
            cam_quat[3],
            cam_quat[0],
            cam_quat[1],
            cam_quat[2],
        ]
        print("camera pose:")
        print(self.env_thick.sim.model.cam_pos[cam_id])
        print(self.env_thick.sim.model.cam_quat[cam_id])
        return camTbase

    def read_target_image(self):
        if self.target_img is None:
            print("Collect target image before MPC first!")
            return
        self.target_img = self._img_transform(self.target_img).to(self.device)

    def collect_target_img(self, eef_target):
        """ set up the scene and collect goal image """
        if len(eef_target) == 2:
            eef_target = [*eef_target, PUSH_HEIGHT, 0, 1, 0, 0]
        else:
            assert len(eef_target) == 7

        control_result = self.control_client.send_target_eef_request(eef_target)
        input("Move the object to the GOAL position. Press Enter to continue...")
        self.target_img = np.copy(self.img)
        self.target_eef = np.array(control_result.end_pose)

        self.target_qpos = control_result.joint_angles

    def go_to_start_pose(self, eef_start):
        """ set up the starting scene """
        result = self.control_client.send_target_eef_request([*eef_start, PUSH_HEIGHT, 0,1,0,0])

        input("Move the object close to the EEF. Press Enter to continue...")
        self.start_img = np.copy(self.img)
        self.start_img = self._img_transform(self.start_img).to(self.device)
        self.start_img = self.start_img.cpu().clamp_(0, 1).numpy()
        self.start_img = np.transpose(self.start_img, axes=(1, 2, 0))
        self.start_img = np.uint8(self.start_img * 255)

    def get_state(self) -> State:
        """Get the current State (eef, qpos, img) of the robot
        Returns:
            State: A namedtuple of current img, eef, qpos
        """
        img = np.copy(self.img)
        img = self._img_transform(img).to(self.device)
        img = img.cpu().clamp_(0, 1).numpy()
        img = np.transpose(img, axes=(1, 2, 0))
        img = np.uint8(img * 255)
        # TODO: change franka control server to return current state if target pose is empty
        control_result = self.control_client.send_target_eef_request([])
        state = State(
            img=img,
            state=[*control_result.end_pose[:3],0,0],
            qpos=control_result.joint_angles,
        )
        return state

    def create_start_goal(self) -> Tuple[State, DemoGoalState]:
        self.read_target_image()
        goal_visual = self.target_img.cpu().clamp_(0, 1).numpy()
        goal_visual = np.transpose(goal_visual, axes=(1, 2, 0))
        self.goal_visual = goal_visual = np.uint8(goal_visual * 255)

        start_visual = self.start_img
        imageio.imwrite(
            os.path.join(self.config.log_dir, "start_goal.png"),
            np.concatenate([start_visual, goal_visual], 1),
        )
        control_result = self.control_client.send_target_eef_request([])
        start = State(
            img=start_visual,
            state=[*control_result.end_pose[:3], 0, 0],
            qpos=control_result.joint_angles,
        )

        mask = self.env_thick.generate_masks([self.target_qpos])[0]

        imageio.imwrite(
            os.path.join(self.config.log_dir, "goal_mask.png"), np.uint8(mask) * 255
        )

        mask = (self._img_transform(mask).type(torch.bool).type(torch.float32)).to(
            self.device
        )

        goal = DemoGoalState(imgs=[goal_visual], masks=[mask])

        return start, goal

    def cem(self, start: State, goal: DemoGoalState, step=0, opt_traj=None):
        actions = self.policy.get_action(start, goal, 0, step, opt_traj)
        return actions

    def execute_action(self, action):
        control_result = self.control_client.send_target_eef_request([])
        end_xy = [
            control_result.end_pose[0] + action[0],
            control_result.end_pose[1] + action[1],
        ]
        control_result = self.control_client.send_target_eef_request(
            target_pose=[*end_xy, PUSH_HEIGHT, 0, 1, 0, 0]
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
        imageio.mimwrite("figures/open_loop.gif", gif, fps=2)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = create_parser()
    parser.add_argument("--execute_optimal_traj", type=str2bool, default=False)
    parser.add_argument("--new_camera_calibration", type=str2bool, default=False)
    parser.add_argument("--save_start_goal", type=str2bool, default=False)
    parser.add_argument("--load_start_goal", type=str, default=None)
    parser.add_argument("--push_type", type=str, default="forward")
    parser.add_argument("--object", type=str, default=" ")

    cf, unparsed = parser.parse_known_args()
    assert len(unparsed) == 0, unparsed
    cf.device = device
    cf.experiment = "control_franka"

    push_type = cf.push_type
    dynamics_model = "vanilla"
    if "roboaware" in cf.dynamics_model_ckpt:
        dynamics_model = "roboaware"
    curr_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    cf.log_dir = os.path.join(
        cf.log_dir,
        push_type + "_" + cf.object + "_" + dynamics_model + "_" + cf.reward_type,
        "debug_cem_" + curr_time,
    )

    cf.debug_cem = True
    # cf.cem_init_std = 0.015
    # cf.action_candidates = 300
    cf.goal_img_with_wrong_robot = True  # makes the robot out of img by pointing up
    cf.cem_open_loop = False
    cf.max_episode_length = 3  # ep length of closed loop execution

    # Initializes a rospy node so that the SimpleActionClient can
    # publish and subscribe over ROS.
    rospy.init_node("franka_visual_mpc_client")

    vmpc = Visual_MPC(config=cf)
    vmpc.control_client.reset()

    if cf.goal_img_with_wrong_robot:
        eef_start_pos = np.array(START_POS[push_type])
        eef_target_pos = [0.55, 0.0, 0.45, 0, 1, 0, 0]
    else:
        eef_target_pos = [0.65, 0]
        eef_start_pos = [eef_target_pos[0], eef_target_pos[1] - start_offset]

    if cf.load_start_goal is not None:
        vmpc.go_to_start_pose(eef_start=eef_start_pos)
        with open(cf.load_start_goal, "rb") as f:
            start, goal = pickle.load(f)
        input("is the start scene ready?")
        vmpc.goal_visual = goal.imgs[0]
    else:
        vmpc.collect_target_img(eef_target_pos)
        vmpc.go_to_start_pose(eef_start=eef_start_pos)
        start, goal = vmpc.create_start_goal()
        if cf.save_start_goal:
            start_goal_file = input("name of start goal pkl file:")
            with open(start_goal_file, "wb") as f:
                pickle.dump([start, goal], f)

    if cf.execute_optimal_traj:
        # push towards camera
        print("executing optimal trajectory")
        actions = [[0.05, 0], [0.05, 0], [0.05, 0], [0.05, 0]]
        vmpc.execute_open_loop(actions)
        sys.exit()

    if cf.cem_open_loop:
        actions = vmpc.cem(start, goal)
        print(actions)
        input("execute actions?")
        vmpc.execute_open_loop(actions)
    else:
        dist = 0.03
        opt_traj = torch.tensor([[dist, 0]] * (cf.horizon - 1))
        if push_type == "left":
            opt_traj = torch.tensor([[0, dist]] * (cf.horizon - 1))
        elif push_type == "right":
            opt_traj = torch.tensor([[0, -dist]] * (cf.horizon - 1))
        img_goal = np.concatenate([start.img, vmpc.goal_visual], 1)
        gif = [img_goal]
        for t in range(cf.max_episode_length):
            act = vmpc.cem(start, goal, t, opt_traj)[0]  # get first action
            print(f"t={t}, executing {act}")
            vmpc.execute_action(act)
            start = vmpc.get_state()
            img_goal = np.concatenate([start.img, vmpc.goal_visual], 1)
            gif.append(img_goal)
        imageio.mimwrite(cf.log_dir + "/closed_loop.gif", gif, fps=2)
