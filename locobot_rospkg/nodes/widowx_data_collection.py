#! /usr/bin/env python

from __future__ import print_function
import random
import cv2
from cv_bridge import CvBridge
import numpy as np
import pathlib
import h5py
from pyquaternion import Quaternion
import math
import interbotix_common_modules.angle_manipulation as ang


# import apriltag
from pupil_apriltags import Detector
import time
from time import gmtime, strftime
import requests

# ROS
import rospy
from sensor_msgs.msg import Image


# Defined by us
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from eef_control.msg import (
    PoseControlAction,
)
from interbotix_xs_modules.arm import InterbotixArmXSInterface, InterbotixArmXSInterface, InterbotixRobotXSCore, InterbotixGripperXSInterface
from interbotix_xs_msgs.msg import JointGroupCommand
import modern_robotics as mr

# TODO: analytical IK for widowX

LOG_DATA = True
PACKAGE_PATH = pathlib.Path(__file__).parent.parent.absolute().__str__()
NUM_TRAJ_TO_COLLECT = 1000
USE_PREPLAN = True

WS_MIN = [0.22, -0.2, 0.15]
WS_MAX = [0.45, 0.2, 0.25]
PUSH_HEIGHT = 0.09
DEFAULT_PITCH = 1.3
DEFAULT_ROLL = 0.0

""" Redistributing figure
"""

policy = {
    "adim": 5,
    "action_order": None,
    "nactions": 30,  # 30 in RoboNet
    "repeat": 1,
    "initial_std": 0.035,  # std dev. in xy
    "initial_std_lift": 0.05,  # std dev. in z
    "initial_std_rot": np.pi / 18,
    "initial_std_grasp": 2.0,
}


def send_msg_to_slack(msg):
    print("sending msg...")
    url = "https://hooks.slack.com/services/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"
    payload = '{"text":"' + msg + '!"}'
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}
    r = requests.post(url, data=payload, headers=headers)


def get_cartesian_pose(bot, matrix=False):
    # Returns cartesian end-effector pose
    pose = bot.arm.get_ee_pose()
    if matrix:
        return pose
    else:
        # 7D pose
        return np.concatenate(
            [pose[:3, -1], np.array(Quaternion(matrix=pose[:3, :3]).elements)]
        )


def eef_control_client(bot, target_pose=[0.3, 0.0, 0.2, 1.3, 0.0]):
    if len(target_pose) >= 5:
        bot.arm.set_ee_pose_components(
            x=target_pose[0],
            y=target_pose[1],
            z=target_pose[2],
            pitch=target_pose[3],
            roll=target_pose[4],
        )
    end_pose = get_cartesian_pose(bot)
    res = PoseControlAction().action_result.result
    res.end_pose = end_pose[:3]
    res.joint_angles = [
        bot.arm.core.joint_states.position[bot.arm.core.js_index_map[name]]
        for name in bot.arm.group_info.joint_names
    ]

    return res


def planar_push_A_to_B(bot, A, B):
    control_result = eef_control_client(bot, target_pose=[])

    # Lift up the eef
    if control_result is not None:
        lift_pose = [*control_result.end_pose, DEFAULT_PITCH, DEFAULT_ROLL]
        lift_pose[2] = PUSH_HEIGHT + 0.1
        control_result = eef_control_client(bot, target_pose=lift_pose)
    else:
        print("ERROR! control server returns None")

    # Move to above start position
    if len(A) == 2:
        control_result = eef_control_client(
            bot, target_pose=[*A, PUSH_HEIGHT + 0.2, DEFAULT_PITCH, DEFAULT_ROLL]
        )
        # Move down
        control_result = eef_control_client(
            bot, target_pose=[*A, PUSH_HEIGHT, DEFAULT_PITCH, DEFAULT_ROLL]
        )
    elif len(A) == 5:
        A[2] = PUSH_HEIGHT + 0.1
        control_result = eef_control_client(bot, target_pose=A)
        # Move down
        A[2] = PUSH_HEIGHT
        control_result = eef_control_client(bot, target_pose=A)

    # Move to end position
    if len(B) == 2:
        control_result = eef_control_client(
            bot, target_pose=[*B, PUSH_HEIGHT, DEFAULT_PITCH, DEFAULT_ROLL]
        )
    elif len(B) == 5:
        control_result = eef_control_client(bot, target_pose=B)


def random_planar_straight_push(client):
    start_xy = [
        random.uniform(WS_MIN[0], WS_MAX[0]),
        random.uniform(WS_MIN[1], WS_MAX[1]),
    ]
    end_xy = [
        random.uniform(WS_MIN[0], WS_MAX[0]),
        random.uniform(WS_MIN[1], WS_MAX[1]),
    ]
    planar_push_A_to_B(client, start_xy, end_xy)


def continuous_planar_pushes(client):
    end_xy = [
        random.uniform(WS_MIN[0], WS_MAX[0]),
        random.uniform(WS_MIN[1], WS_MAX[1]),
    ]
    control_result = eef_control_client(
        client, target_pose=[*end_xy, PUSH_HEIGHT, DEFAULT_PITCH, DEFAULT_ROLL]
    )


def construct_initial_sigma(hp, adim, t=None):
    xy_std = hp["initial_std"]
    diag = [xy_std ** 2, xy_std ** 2]

    if hp["action_order"] is not None:
        diag = []
        for a in hp["action_order"]:
            if a == "x" or a == "y":
                diag.append(xy_std ** 2)
            elif a == "z":
                diag.append(hp["initial_std_lift"] ** 2)
            elif a == "theta":
                diag.append(hp["initial_std_rot"] ** 2)
            elif a == "grasp":
                diag.append(hp["initial_std_grasp"] ** 2)
            else:
                print("Not implemented")
    else:
        if adim >= 3:
            diag.append(hp["initial_std_lift"] ** 2)
        if adim >= 4:
            diag.append(hp["initial_std_rot"] ** 2)
        if adim == 5:
            diag.append(hp["initial_std_grasp"] ** 2)

    adim = len(diag)
    diag = np.tile(diag, hp["nactions"])
    diag = np.array(diag)

    if "reduce_std_dev" in hp:
        assert "reuse_mean" in hp
        if t >= 2:
            print("reducing std dev by factor", hp["reduce_std_dev"])
            # reducing all but the last repeataction in the sequence since it can't be reused.
            diag[: (hp["nactions"] - 1) * adim] *= hp["reduce_std_dev"]

    sigma = np.diag(diag)
    return sigma


def process_action(action, state):
    """
    If state + action is out of boundary, reverse action
    state: current eef position
    """
    output_action = np.copy(action)
    # when input action drives the eef out of bound, revert it
    if len(state) < 2:
        print("Warning: input state has incorrect shape:", state)
        return output_action

    end_pos = state[0:2] + action[:2]
    # if end_pos[0] < 0.25 and end_pos[1] > -0.2 and end_pos[1] < 0.2:
    #     print("Warning: input action drives the eef self collision, revert it")
    #     output_action = -action
    if (
        # end_pos[1] > 0.52 - end_pos[0]
        # or end_pos[1] < end_pos[0] - 0.52
        # or end_pos[1] > end_pos[0] - 0.03
        # or end_pos[1] < -end_pos[0] + 0.03
        not (0.25 < end_pos[0] < 0.5) or
        not (-0.25 < end_pos[1] < 0.25)
    ):
        print("Warning: input action drives the eef out of bound, revert it")
        output_action = -action
    return output_action


def gaussian_push():
    """
    adim: Environment's action dimension, 5
    sdim: Environment's state dimension, 5
    """
    mean = np.zeros(policy["adim"] * policy["nactions"])
    # initialize mean and variance of the discrete actions to their mean and variance used during data collection
    sigma = construct_initial_sigma(policy, policy["adim"])
    actions = np.random.multivariate_normal(mean, sigma).reshape(policy["nactions"], -1)
    # actions = process_actions(actions, state, bound)
    return actions


def temporal_gaussian_push(beta=0.8):
    """
    Temporally correlated gaussian push
    adim: Environment's action dimension, 5
    sdim: Environment's state dimension, 5
    """
    mean = np.zeros(policy["adim"] * policy["nactions"])
    # initialize mean and variance of the discrete actions to their mean and variance used during data collection
    sigma = construct_initial_sigma(policy, policy["adim"])
    actions = np.random.multivariate_normal(mean, sigma).reshape(policy["nactions"], -1)
    new_actions = np.zeros_like(actions)
    new_actions[0] = actions[0]
    for i in range(1, len(new_actions)):
        new_actions[i] = (
            beta * actions[i] + (1 - beta) * actions[i - 1]
        )  # actions = process_actions(actions, state, bound)

    return actions


def preplan_trajectory(init_state, actions):
    curr_state = np.copy(init_state)
    way_points = []
    for t in range(actions.shape[0]):
        out_action = process_action(actions[t], curr_state)
        actions[t] = out_action
        curr_state = np.array(
            [curr_state[0] + actions[t, 0], curr_state[1] + actions[t, 1], PUSH_HEIGHT]
        )
        way_points.append(curr_state)
    way_points = np.stack(way_points)
    return way_points, actions


def redistribute_objects(bot):
    print("Reseting objects...")
    # +/- x - robot front / back
    # +/- y - robot left / right
    # +/- z - robot up / down

    # reset bottom left
    eef_control_client(bot, target_pose=[0.17, 0.17, 0.15, 1.3, -1.4])
    eef_control_client(bot, target_pose=[0.17, 0.17, 0.075, 1.3, -1.4])
    eef_control_client(bot, target_pose=[0.3, 0.0, 0.1, 1.3, -1.4])

    # reset side left
    eef_control_client(bot, target_pose=[0.38, 0.25, 0.3, 0.8, -1])
    eef_control_client(bot, target_pose=[0.38, 0.25, 0.1, 1.3, -1])
    eef_control_client(bot, target_pose=[0.3, 0.0, 0.1, 1.3, -1])

    # reset top left
    eef_control_client(bot, target_pose=[0.5, 0.25, 0.3, 0.5, 0.0])
    eef_control_client(bot, target_pose=[0.5, 0.25, 0.1, 0.5, 0.0])
    eef_control_client(bot, target_pose=[0.4, 0.1, 0.1, 1.3, 0.0])

    # reset top middle
    eef_control_client(bot, target_pose=[0.5, 0, 0.3, 0.5, 0.0])
    eef_control_client(bot, target_pose=[0.5, 0, 0.1, 0.5, 0.0])
    eef_control_client(bot, target_pose=[0.4, 0.0, 0.1, 1.3, 0.0])

    # # reset top right
    eef_control_client(bot, target_pose=[0.5, -0.25, 0.3, 0.5, 0.0])
    eef_control_client(bot, target_pose=[0.5, -0.25, 0.1, 0.5, 0.0])
    eef_control_client(bot, target_pose=[0.4, -0.1, 0.1, 1.3, 0.0])

    # # reset side right
    eef_control_client(bot, target_pose=[0.38, -0.25, 0.3, 0.8, 1])
    eef_control_client(bot, target_pose=[0.38, -0.25, 0.1, 1.3, 1])
    eef_control_client(bot, target_pose=[0.3, 0.0, 0.1, 1.3, 1])

    # reset bottom right
    eef_control_client(bot, target_pose=[0.17, -0.17, 0.15, 1.3, 1.4])
    eef_control_client(bot, target_pose=[0.17, -0.17, 0.075, 1.3, 1.4])
    eef_control_client(bot, target_pose=[0.3, 0.0, 0.1, 1.3, 1.4])

    # # reset bottom middle
    eef_control_client(bot, target_pose=[0.1, 0, 0.15, 1.3, 0.0])
    eef_control_client(bot, target_pose=[0.1, 0, 0.075, 1.3, 0.0])
    eef_control_client(bot, target_pose=[0.3, 0.0, 0.1, 1.3, 0.0])



class Data_Collector(object):
    def __init__(self):
        self.bot = InterbotixManipulatorXS("wx250s", "arm", "gripper")
        self.img_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.img_callback
        )
        self.depth_sub = rospy.Subscriber(
            "/camera/depth/image_rect_raw", Image, self.depth_callback
        )
        self.cv_bridge = CvBridge()
        self.img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth = np.zeros((480, 640), dtype=np.uint16)

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
            camera_params=[387.028, 387.028, 318.089, 242.104],
            tag_size=0.0286,
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

    def check_qpos_eef_match(self, eef, qpos):
        qpos_from_eef = np.zeros(5)
        # qpos_from_eef[0:4] = self.ik_solver.ik(
        #     eef, alpha=-DEFAULT_PITCH, cur_arm_config=qpos
        # )
        # qpos_from_eef[4] = DEFAULT_ROLL

        # qpos_diff = np.abs(qpos - qpos_from_eef)
        # print("qpos diff:       ", qpos_diff)
        # print("logged qpos:     ", qpos)
        # print("computed qpos:   ", qpos_from_eef)

    def data_collection(self):
        time.sleep(0.1)
        eef_control_client(self.bot, target_pose=[0.3, 0.0, 0.2, 1.3, 0.0])

        for _ in range(1):
            self.get_camera_pose_from_apriltag()

        send_msg_to_slack("Start collecting trajectory...")
        for traj_num in range(NUM_TRAJ_TO_COLLECT):
            if traj_num % 20 == 0:
                send_msg_to_slack("Collecting trajectory " + str(traj_num))

            # actions = gaussian_push()
            actions = gaussian_push()
            control_result = eef_control_client(self.bot, target_pose=[])
            if control_result is None:
                print("ERROR! control server returns None")
                send_msg_to_slack(
                    "<@U01E9UV9GMT> <@U015A40J7S6> <@U015BDN6NG3> ERROR! control server returns None"
                )
                return
            elif (
                len(control_result.end_pose) != 3
                or len(control_result.joint_angles) != 6
            ):
                send_msg_to_slack(
                    "<@U01E9UV9GMT> <@U015A40J7S6> <@U015BDN6NG3> Warning! control_result.end_pose has incorrect shape, skip this trajectory"
                )
                redistribute_objects(self.bot)
                continue

            self.check_qpos_eef_match(
                np.array(control_result.end_pose), np.array(control_result.joint_angles)
            )

            skip = False

            observations = []
            depths = []
            qpos = []
            eef_states = []

            # log initial state information
            eef_states.append(control_result.end_pose)
            qpos.append(control_result.joint_angles)
            observations.append(self.img)
            depths.append(self.depth)

            if USE_PREPLAN:
                way_points, actions = preplan_trajectory(
                    control_result.end_pose, actions
                )
                for t in range(way_points.shape[0]):
                    control_result = eef_control_client(
                        self.bot,
                        target_pose=[
                            way_points[t, 0],
                            way_points[t, 1],
                            PUSH_HEIGHT,
                            DEFAULT_PITCH,
                            DEFAULT_ROLL,
                        ],
                    )
                    if control_result is None:
                        print("ERROR! control server returns None")
                        send_msg_to_slack(
                            "<@U01E9UV9GMT> <@U015A40J7S6> <@U015BDN6NG3> ERROR! control server returns None"
                        )
                        return
                    elif (
                        len(control_result.end_pose) != 3
                        or len(control_result.joint_angles) != 6
                    ):
                        send_msg_to_slack(
                            "<@U01E9UV9GMT> <@U015A40J7S6> <@U015BDN6NG3> Warning! control_result.end_pose has incorrect shape, skip this trajectory"
                        )
                        redistribute_objects(self.bot)
                        skip = True
                        break

                    self.check_qpos_eef_match(
                        np.array(control_result.end_pose),
                        np.array(control_result.joint_angles),
                    )

                    eef_states.append(control_result.end_pose)
                    qpos.append(control_result.joint_angles)
                    observations.append(self.img)
                    depths.append(self.depth)

                if skip:
                    continue
            else:
                for t in range(actions.shape[0]):
                    out_action = process_action(actions[t], control_result.end_pose)
                    actions[t] = out_action
                    end_xy = [
                        control_result.end_pose[0] + actions[t, 0],
                        control_result.end_pose[1] + actions[t, 1],
                    ]
                    control_result = eef_control_client(
                        self.bot,
                        target_pose=[*end_xy, PUSH_HEIGHT, DEFAULT_PITCH, DEFAULT_ROLL],
                    )
                    if control_result is None:
                        print("ERROR! control server returns None")
                        send_msg_to_slack(
                            "<@U01E9UV9GMT> <@U015A40J7S6> <@U015BDN6NG3> ERROR! control server returns None"
                        )
                        return
                    elif (
                        len(control_result.end_pose) != 3
                        or len(control_result.joint_angles) != 6
                    ):
                        send_msg_to_slack(
                            "<@U01E9UV9GMT> <@U015A40J7S6> <@U015BDN6NG3> Warning! control_result.end_pose has incorrect shape, skip this trajectory"
                        )
                        redistribute_objects(self.bot)
                        skip = True
                        break

                    eef_states.append(control_result.end_pose)
                    qpos.append(control_result.joint_angles)
                    observations.append(self.img)
                    depths.append(self.depth)

                if skip:
                    continue

            # TODO: log camera calibration
            # The time here is EDT + 4h
            curr_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
            print(curr_time)
            if LOG_DATA:
                hf_file = h5py.File(
                    PACKAGE_PATH + "/data/data_" + curr_time + ".hdf5", "w"
                )

                observations = np.stack(observations)
                hf_file.create_dataset("observations", data=observations)

                depths = np.stack(depths)
                hf_file.create_dataset("depths", data=depths)

                # qpos is 5D: from base to elbow, gripper motor not included
                qpos = np.stack(qpos)
                hf_file.create_dataset("qpos", data=qpos)

                eef_states = np.stack(eef_states)
                hf_file.create_dataset("states", data=eef_states)

                # actions are 5d, where last three elements are zeros
                print("action logged:", actions.shape)
                actions[:, 2:] = 0
                hf_file.create_dataset("actions", data=actions)
                hf_file.close()

            if traj_num % 7 == 0:
                redistribute_objects(self.bot)


def move_along_viewpoint_path(bot):
    bot.arm.set_ee_pose_components(x=0.35, z=0.2, pitch=0.8)

    for _ in range(5):
        # set_ee_cartesian_trajectory takes delta
        bot.arm.set_ee_cartesian_trajectory(x=-0.05, z=-0.1, pitch=-0.8, moving_time=15)
        bot.arm.set_ee_cartesian_trajectory(x=0.05, z=0.1, pitch=0.8, moving_time=15)

    bot.arm.go_to_home_pose()
    bot.arm.go_to_sleep_pose()

if __name__ == "__main__":
    dc = Data_Collector()
    # eef_control_client(dc.bot, target_pose=[0.25, 0.17, 0.15, 1.3, 0])
    dc.data_collection()
