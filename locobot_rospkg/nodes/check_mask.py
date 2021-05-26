#! /usr/bin/env python

from __future__ import print_function

import pathlib
import time
from time import gmtime, strftime

import actionlib
from eef_control.msg import *
import cv2
import h5py
import numpy as np
import rospy
from cv_bridge import CvBridge
from locobot_rospkg.nodes.data_collection_client import (DEFAULT_PITCH,
                                                         DEFAULT_ROLL,
                                                         PUSH_HEIGHT,
                                                         eef_control_client)
from pupil_apriltags import Detector
from scipy.spatial.transform.rotation import Rotation
from sensor_msgs.msg import Image
from src.env.robotics.masks.locobot_mask_env import LocobotMaskEnv

PACKAGE_PATH = pathlib.Path(__file__).parent.parent.absolute().__str__()


class MaskChecker(object):
    def __init__(self):
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

        self.env = LocobotMaskEnv()
        camTbase = self.get_cam_calibration()
        self.set_camera_calibration(camTbase)


    def img_callback(self, data):
        self.img = self.cv_bridge.imgmsg_to_cv2(data)

    def depth_callback(self, data):
        self.depth = self.cv_bridge.imgmsg_to_cv2(data)

    def get_cam_calibration(self):
        control_result = eef_control_client(
            self.control_client,
            target_pose=[0.35, 0, PUSH_HEIGHT, DEFAULT_PITCH, DEFAULT_ROLL],
        )
        time.sleep(5)
        control_result = eef_control_client(
            self.control_client,
            target_pose=[],
        )
        print(control_result.end_pose)
        # tag to camera transformation
        pose_t, pose_R = self.get_camera_pose_from_apriltag()
        if pose_t is None or pose_R is None:
            return None
        # TODO: figure out qpos / end effector accuracy
        # currently qpos is better than using end effector
        target_qpos = control_result.joint_angles
        self.env.sim.data.qpos[self.env._joint_references] = target_qpos
        self.env.sim.forward()

        # tag to base transformation
        tagTbase = np.column_stack(
            (
                self.env.sim.data.get_geom_xmat("ar_tag_geom"),
                self.env.sim.data.get_geom_xpos("ar_tag_geom"),
            )
        )
        tagTbase = np.row_stack((tagTbase, [0, 0, 0, 1]))

        tagTcam = np.column_stack((pose_R, pose_t))
        tagTcam = np.row_stack((tagTcam, [0, 0, 0, 1]))

        # tag in camera to tag in robot transformation
        # For explanation, refer to Kun's hand drawing
        tagcTtagw = np.array(
            [[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
        )

        camTbase = tagTbase @ tagcTtagw @ np.linalg.inv(tagTcam)
        print("camera2world:")
        print(camTbase)
        return camTbase

    def set_camera_calibration(self, camTbase):
        rot_matrix = camTbase[:3, :3]
        cam_pos = camTbase[:3, 3]
        rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
        cam_rot = Rotation.from_matrix(rot_matrix) * rel_rot

        cam_id = 0
        # offset = [0, -0.007, 0.02]
        offset = [0, -0.015, 0.0125]
        print("set camera calibration")
        print("applying offset", offset)
        # offset = [0, 0, 0.0]
        self.env.sim.model.cam_pos[cam_id] = cam_pos + offset
        cam_quat = cam_rot.as_quat()
        self.env.sim.model.cam_quat[cam_id] = [
            cam_quat[3],
            cam_quat[0],
            cam_quat[1],
            cam_quat[2],
        ]
        print("camera pose:")
        print(self.env.sim.model.cam_pos[cam_id])
        print(self.env.sim.model.cam_quat[cam_id])
        return camTbase

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

    def show_overlap(self):
        # show the overlap between the mask and current image
        control_result = eef_control_client(self.control_client, target_pose=[])
        qpos = control_result.joint_angles
        mask = self.env.generate_masks([qpos], width=640, height=480)[0]
        # import ipdb; ipdb.set_trace()
        while True:
            cam_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            cam_img[mask] = (0, 255, 255)
            cv2.imshow("overlay", cam_img)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # space
                pass
            elif key == ord("q"):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    # Initializes a rospy node so that the SimpleActionClient can
    # publish and subscribe over ROS.
    rospy.init_node("match_robot_position")

    rpos = MaskChecker()
    rpos.show_overlap()
