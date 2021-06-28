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
from cv_bridge import CvBridge

from eef_control.msg import *
from sensor_msgs.msg import Image

class DemoCollector(object):
    def __init__(self, directory="human_demo"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        # Creates the SimpleActionClient, passing the type of the action
        self.img_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.img_callback
        )
        self.cv_bridge = CvBridge()
        self.img = np.zeros((480, 640, 3), dtype=np.uint8)

    def img_callback(self, data):
        self.img = self.cv_bridge.imgmsg_to_cv2(data)

    def collect_demo(self, demo_name="test"):
        imgs = []
        i = 0
        while True:
            x = input("Type s to save an image or q to quit")
            if x == "s":
                imgs.append(np.copy(self.img))
                i += 1
                print(f"Stored img {i}")
            elif x == "q":
                break
            else:
                print(f"{x} is not a valid input.")
        pkl_path = os.path.join(self.directory, demo_name + ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(imgs, f)
        gif_path = os.path.join(self.directory, demo_name + ".gif")
        imageio.mimwrite(gif_path, imgs)


def demo_main():
    rospy.init_node("franka_demonstration_collection")
    collector = DemoCollector()
    collector.collect_demo()

if __name__ == "__main__":
    # 1. Collect demos into pkl files
    # demo_main()

    # 2. dump into pngs for labeling
    # from PIL import Image
    # path = "human_demo/multipush_2/test.pkl"
    # with open(path, "rb") as f:
    #     imgs = pickle.load(f)
    # for i, im in enumerate(imgs):
    #     im = Image.fromarray(im)
    #     im.save(os.path.join("human_demo", f"{i}.png"))


    # check labeled masks
    from PIL import Image as PILImage
    folder = "human_demo/multipush_2"
    imgs = []
    for i in range(6):
        img_path =os.path.join(folder, f"{i}.png")
        img = np.array(PILImage.open(img_path))
        mask_path = os.path.join(folder, f"{i}mask.npy")
        mask = np.load(mask_path)
        mask[mask != 0] = 255
        # convert to boolean mask
        mask = mask.astype(bool)
        bool_mask_path = os.path.join(folder, f"{i}mask.pkl")
        with open(bool_mask_path, "wb") as f:
            pickle.dump(mask, f)

        img[mask] = 255
        imgs.append(img)
    imageio.mimwrite(os.path.join(folder,"test_masks.gif"), imgs, fps=2)