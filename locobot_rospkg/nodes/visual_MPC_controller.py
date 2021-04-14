#! /usr/bin/env python

from __future__ import print_function
import random
import cv2
from cv_bridge import CvBridge
import numpy as np
import pathlib
import h5py
from pupil_apriltags import Detector
import time
from time import gmtime, strftime
import requests

# ROS
import rospy
import actionlib
from sensor_msgs.msg import Image

# Defined by us
from eef_control.msg import *

class Visual_MPC(object):
    def __init__(self):
        # Creates the SimpleActionClient, passing the type of the action
        self.control_client = actionlib.SimpleActionClient(
            'eef_control', eef_control.msg.PoseControlAction)

        self.img_sub = rospy.Subscriber("/camera/color/image_raw",
                                        Image,
                                        self.img_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw",
                                          Image,
                                          self.depth_callback)
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
            detector = Detector(families='tag36h11',
                                nthreads=1,
                                quad_decimate=1.0,
                                quad_sigma=0.0,
                                refine_edges=1,
                                decode_sharpening=0.25,
                                debug=0)

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        results = []
        results = detector.detect(gray,
                                  estimate_tag_pose=True,
                                  camera_params=[612.45,
                                                 612.45,
                                                 330.55,
                                                 248.61],
                                  tag_size=0.0353)
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

    def rollout(self):
        pass

if __name__ == '__main__':
    # Initializes a rospy node so that the SimpleActionClient can
    # publish and subscribe over ROS.
    rospy.init_node('visual_mpc_client')

    vmpc = Visual_MPC()
    vmpc.rollout()
