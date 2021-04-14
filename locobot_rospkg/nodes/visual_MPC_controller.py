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
from tqdm import trange
import ipdb

# ROS
import rospy
import actionlib
from sensor_msgs.msg import Image

# Learning
import torch

# Defined by us
from eef_control.msg import *
from locobot_rospkg.nodes.data_collection_client import eef_control_client

from src.config import argparser
from src.prediction.models.dynamics import SVGConvModel
from src.dataset.locobot.locobot_model import LocobotAnalyticalModel
from src.utils.plot import save_gif


class Visual_MPC(object):
    def __init__(self, device="cuda"):
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

        self.device = device

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

    def read_target_image(self, path):
        pass

    def load_model(self, config, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = SVGConvModel(config).to(self.device)
        self.model.load_state_dict(ckpt["model"])

    def rollout(self):
        pass


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf, _ = argparser()
    cf.device = device

    CKPT_PATH = "/home/pallab/locobot_ws/src/roboaware/checkpoints/locobot_689_ckpt_213000.pt"

    # Initializes a rospy node so that the SimpleActionClient can
    # publish and subscribe over ROS.
    rospy.init_node('visual_mpc_client')

    vmpc = Visual_MPC()
    vmpc.load_model(cf, ckpt_path=CKPT_PATH)
    vmpc.get_camera_pose_from_apriltag()

    vmpc.rollout()
