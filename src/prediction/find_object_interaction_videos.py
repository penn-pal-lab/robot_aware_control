""" Find videos with object interaction in the dataset """
from collections import defaultdict
from math import floor
import os
from pdb import main
import pickle
import random
from dataclasses import dataclass

from torch.nn.modules import loss
from src.prediction.losses import world_mse_criterion
from src.prediction.models.dynamics import CopyModel
from src.config import argparser
from time import time
from typing import Any

import cv2
import h5py
import imageio
import ipdb
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from ipdb import set_trace as st
from src.utils.camera_calibration import world_to_camera_dict
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import has_file_allowed_extension
from tqdm import trange
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt


class RobotDataset(data.Dataset):
    def __init__(
        self, hdf5_list, robot_list, config, augment_img=False, load_snippet=False
    ) -> None:
        """
        hdf5_list: list of hdf5 files to load
        robot_list: list of robot type for each hdf5 file
        split: load a random snippet, or the entire video depending on split.
        """
        self._traj_names = hdf5_list
        self._traj_robots = robot_list
        self._config = config
        self._data_root = config.data_root

        self._video_length = config.video_length
        if load_snippet:
            self._video_length = config.n_past + config.n_future
        self._action_dim = config.action_dim
        self._impute_autograsp_action = config.impute_autograsp_action
        self._augment_img = augment_img
        self._img_transform = tf.Compose(
            [tf.ToTensor(), tf.CenterCrop(config.image_width-5), tf.Resize(config.image_width)]
        )
        # if self._augment_img:
        #     r = config.color_jitter_range
        #     self._crop_resize = tf.Compose(
        #         [
        #             tf.RandomCrop(config.random_crop_size),
        #             tf.Resize(config.image_width),
        #         ]
        #     )
        #     self._jitter_color = tf.ColorJitter(r,r,r,r)

        self._rng = np.random.RandomState(config.seed)
        self._memory = {}
        if config.preload_ram:
            self.preload_ram()

    def preload_ram(self):
        # load everything into memory
        for i in trange(len(self._traj_names), desc=f"loading into RAM"):
            self._memory[i] = self.__getitem__(i)

    def __getitem__(self, idx):
        """
        Opens the hdf5 file and extracts the videos, actions, states, and masks
        """
        if idx in self._memory:
            return self._memory[idx]

        robonet_root = self._data_root
        name = self._traj_names[idx]
        hdf5_path = os.path.join(robonet_root, name)
        with h5py.File(hdf5_path, "r") as hf:
            ep_len = hf["frames"].shape[0]
            assert ep_len >= self._video_length, f"{ep_len}, {hdf5_path}"
            start = 0
            end = ep_len
            if ep_len > self._video_length:
                offset = ep_len - self._video_length
                start = self._rng.randint(0, offset + 1)
                end = start + self._video_length

            images = hf["frames"][start:end]
            states = hf["states"][start:end].astype(np.float32)
            # actions = hf["actions"][start:end - 1].astype(np.float32)
            low = hf["low_bound"][:]
            high = hf["high_bound"][:]
            actions = self._load_actions(hf, low, high, start, end - 1)
            masks = hf["mask"][start:end].astype(np.float32)

            assert (
                len(images) == len(states) == len(actions) + 1 == len(masks)
            ), f"{hdf5_path}, {images.shape}, {states.shape}, {actions.shape}, {masks.shape}"

            # preprocessing
            images, masks = self._preprocess_images_masks(images, masks)
            actions = self._preprocess_actions(states, actions, low, high, idx)
            states = self._preprocess_states(states, low, high)

            robot = hf.attrs["robot"]

        out = {
            "images": images,
            "states": states,
            "actions": actions,
            "masks": masks,
            "robot": robot,
            "file_name": name,
            "file_path": hdf5_path,
            "idx": idx
        }
        return out

    def _load_actions(self, file_pointer, low, high, start, end):
        actions = file_pointer["actions"][:].astype(np.float32)
        a_T, adim = actions.shape[0], actions.shape[1]
        if self._action_dim == adim:
            return actions[start:end]
        elif self._impute_autograsp_action and adim + 1 == self._action_dim:
            action_append, old_actions = (
                np.zeros((a_T, 1)),
                actions,
            )
            next_state = file_pointer["states"][:][1:, -1]
            high_val, low_val = high[-1], low[-1]
            midpoint = (high_val + low_val) / 2.0

            for t, s in enumerate(next_state):
                if s > midpoint:
                    action_append[t, 0] = high_val
                else:
                    action_append[t, 0] = low_val
            new_actions = np.concatenate((old_actions, action_append), axis=-1)
            return new_actions[start:end].astype(np.float32)
        else:
            raise ValueError(f"file adim {adim}, target adim {self._action_dim}")

    def _preprocess_images_masks(self, images, masks):
        """
        Converts numpy image (uint8) [0, 255] to tensor (float) [0, 1].
        """
        if self._augment_img:
            # img_mask = torch.cat([video_tensor, mask_tensor], dim=1)
            crop_size = [self._config.random_crop_size] * 2
            img_width = self._config.image_width
            i, j, th, tw = tf.RandomCrop.get_params(
                torch.zeros(3, img_width, img_width), crop_size
            )
            brightness = (1 - 0.2, 1 + 0.2)
            contrast = (1 - 0.2, 1 + 0.2)
            saturation = (1 - 0.2, 1 + 0.2)
            hue = (-0.1, 0.1)
            jitter = tf.ColorJitter.get_params(brightness, contrast, saturation, hue)
            aug_imgs = []
            aug_masks = []
            for img, mask in zip(images, masks):
                img = self._img_transform(img)
                mask = self._img_transform(mask)
                crop_img = F.crop(img, i, j, th, tw)
                crop_mask = F.crop(mask, i, j, th, tw)
                resized_img = F.resize(crop_img, img_width)
                # cast back to 0 or 1 value
                resized_mask = (
                    F.resize(crop_mask, img_width).type(torch.bool).type(torch.float32)
                )
                color_img = jitter(resized_img)
                aug_imgs.append(color_img)
                aug_masks.append(resized_mask)
            video_tensor = torch.stack(aug_imgs)
            mask_tensor = torch.stack(aug_masks)
        else:
            video_tensor = torch.stack([self._img_transform(i) for i in images])
            mask_tensor = torch.stack([self._img_transform(i) for i in masks])
        return video_tensor, mask_tensor

    def _preprocess_states(self, states, low, high):
        """
        states: T x 5 array of [x,y,z, r, f], r is rotation in radians, f is gripper force
        We only need to normalize the gripper force.
        XYZ is already normalized in the hdf5 file.
        yaw is in radians across all robots, so it is consistent.
        """
        states = torch.from_numpy(states)
        force_min, force_max = low[4], high[4]
        states[:, 4] = (states[:, 4] - force_min) / (force_max - force_min)
        return states

    def _convert_world_to_camera_pos(self, state, w_to_c):
        e_to_w = np.eye(4)
        e_to_w[:3, 3] = state[:3]
        e_to_c = w_to_c @ e_to_w
        pos_c = e_to_c[:3, 3]
        return pos_c

    def _denormalize(self, states, low, high):
        states = states * (high - low)
        states = states + low
        return states

    def _convert_actions(self, states, actions, w_to_c, low, high):
        """
        Concert raw actions to camera frame displacements
        """
        states[:, :3] = self._denormalize(states[:, :3], low[:3], high[:3])
        old_actions = actions.copy()
        for t in range(len(actions)):
            state = states[t]
            pos_c = self._convert_world_to_camera_pos(state, w_to_c)
            next_state = states[t].copy()
            next_state[:4] += old_actions[t][:4]
            next_pos_c = self._convert_world_to_camera_pos(next_state, w_to_c)
            true_offset_c = next_pos_c - pos_c
            actions[t][:3] = true_offset_c

    def _impute_camera_actions(self, states, actions, w_to_c, low, high):
        """
        Just calculate the true offset between states instead of using  the recorded actions.
        """
        states[:, :3] = self._denormalize(states[:, :3], low[:3], high[:3])
        for t in range(len(actions)):
            state = states[t]
            pos_c = self._convert_world_to_camera_pos(state, w_to_c)
            next_state = states[t + 1]
            next_pos_c = self._convert_world_to_camera_pos(next_state, w_to_c)
            true_offset_c = next_pos_c - pos_c
            actions[t][:3] = true_offset_c

    def _impute_true_actions(self, states, actions, low, high):
        """
        Set the action to what happened between states, not recorded actions.
        """
        states[:, :3] = self._denormalize(states[:, :3], low[:3], high[:3])
        for t in range(len(actions)):
            state = states[t][:3]
            next_state = states[t + 1][:3]
            true_offset_c = next_state - state
            actions[t][:3] = true_offset_c

    def _preprocess_actions(self, states, actions, low, high, idx):
        strategy = self._config.preprocess_action
        if strategy == "raw":
            return torch.from_numpy(actions)
        elif strategy == "state_infer":
            states = states.copy()
            self._impute_true_actions(states, actions, low, high)
            return torch.from_numpy(actions)
        # if actions are in camera frame...
        filename = self._traj_names[idx]
        robot_type = self._traj_robots[idx]
        if self._config.training_regime == "multirobot":
            # convert everything to camera coordinates.
            filename = self._traj_names[idx]
            robot_type = self._traj_robots[idx]
            if robot_type == "sawyer":
                # TODO: account for camera config and index
                world2cam = world_to_camera_dict["sawyer_sudri0_c0"]
            elif robot_type == "widowx":
                world2cam = world_to_camera_dict["widowx1"]
            elif robot_type == "baxter":
                arm = "left" if "left" in filename else "right"
                world2cam = world_to_camera_dict[f"baxter_{arm}"]

        elif self._config.training_regime == "singlerobot":
            # train on sawyer, convert actions to camera space
            # TODO: account for camera config and index
            world2cam = world_to_camera_dict["sawyer_sudri0_c0"]
        elif self._config.training_regime == "finetune":
            # finetune on baxter, convert to camera frame
            # Assumes the baxter arm is right arm!
            arm = "left" if "left" in filename else "right"
            world2cam = world_to_camera_dict[f"baxter_{arm}"]

        states = states.copy()
        if strategy == "camera_raw":
            self._convert_actions(states, actions, world2cam, low, high)
        elif strategy == "camera_state_infer":
            self._impute_camera_actions(states, actions, world2cam, low, high)
        return torch.from_numpy(actions)

    def __len__(self):
        return len(self._traj_names)

def eval_step(data, model: CopyModel, cf):
    """Evaluates the error metric on the data"""
    x, masks, names = data
    losses = defaultdict(float)
    for i in range(1, cf.n_past + cf.n_future):
        x_j = x[i-1]
        x_i, m_i = x[i], masks[i]
        x_pred = model(x_j, None, x_i, m_i)
        # compute world loss error
        world_mse = world_mse_criterion(x_pred, x_i, m_i)
        losses["world_mse"] += world_mse
    # for k, v in losses.items():
        # losses[k] = v / (cf.n_past + cf.n_future)
    return losses

def visualize_files():
    config, _ = argparser()
    config.data_root = "/home/ed/Robonet"
    config.batch_size = 1
    config.video_length = 31
    config.image_width = 64
    # config.impute_autograsp_action = True
    config.num_workers = 4
    config.action_dim = 5
    config.n_past = 1
    config.n_future = 5

    # finetune on baxter left data
    file_type = "hdf5"
    files = []
    file_labels = []
    robots = ["baxter"]
    data_path = os.path.join(config.data_root, "new_hdf5")
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            for r in robots:
                if f"{r}_right" in d.path:
                    files.append(d.path)
                    file_labels.append(r)
                    break
    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)

    # only use 500 videos (400 training) like robonet
    files = files[:500]
    file_labels = ["baxter"] * len(files)
    # iterate through videos
    # compute error for each video
    # if error > threshold, save the video
    dataset = RobotDataset(files, file_labels, config)
    train_loader = DataLoader(
        dataset,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    model = CopyModel()
    window = config.n_past + config.n_future
    T = 31
    pbar = tqdm(initial=0, total=len(dataset) * (floor(T/window)), desc="processing videos")
    num_high_err_videos = 0
    num_high_err_snippets = 0
    num_snippets = 0
    all_err = []
    for d in train_loader:
        x = d["images"].transpose_(0, 1) # T x B
        masks = d["masks"].transpose_(0,1) # T x B
        names = d["file_name"] # B
        T = x.shape[0]
        all_losses = defaultdict(float)
        high_err_video = False
        for i in range(floor(T / window)):
            s = i * window
            e = (i + 1) * window
            batch = x[s:e], masks[s:e], names
            losses = eval_step(batch, model, config)
            num_snippets += 1
            if losses["world_mse"] > 0.01:
                high_err_video = True
                num_high_err_snippets += 1

            # if losses["world_mse"] > 0.01:
            #     # save video
            #     mask = batch[1]
            #     robot_mask = mask.type(torch.bool)
            #     robot_mask = robot_mask.repeat(1,1,3,1,1)
            #     batch[0][robot_mask] *= 0
            #     imgs = (255 * batch[0].squeeze().permute(0,2,3,1)).cpu().numpy().astype(np.uint8)
            #     name = names[0].split("/")[-1][:-5]
            #     err = losses["world_mse"]
            #     gif_name = f"{name}_{losses['world_mse']:.6f}.gif"
            #     imageio.mimwrite(gif_name, imgs)
            all_err.append(losses["world_mse"])
            pbar.update()

        if high_err_video:
            num_high_err_videos += 1
    print(f"number of videos with high errors {num_high_err_videos}/{len(dataset)}")
    print(f"number of snippets with high errror: {num_high_err_snippets}/{num_snippets}")

    # histogram of world err
    all_err = np.asarray(all_err)
    print(all_err.mean())
    plt.hist(all_err, bins=24, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(all_err.mean(), color='k', linestyle='dashed', linewidth=1)
    # plt.show()
    plt.savefig("world_error_histogram.png")


if __name__ == "__main__":
    """
    Generate meta dictionary for recording baseline error per video
    """
    config, _ = argparser()
    config.data_root = "/home/ed/Robonet"
    config.batch_size = 1
    config.video_length = 31
    config.image_width = 64
    # config.impute_autograsp_action = True
    config.num_workers = 4
    config.action_dim = 5
    config.n_past = 1
    config.n_future = 5

    # finetune on baxter left data
    file_type = "hdf5"
    files = []
    file_labels = []
    robots = ["baxter"]
    arm = "left"

    data_path = os.path.join(config.data_root, "new_hdf5")
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            for r in robots:
                if f"{r}_{arm}" in d.path:
                    files.append(d.path)
                    file_labels.append(r)
                    break
    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)

    # only use 500 videos (400 training) like robonet
    files = files[:]
    file_labels = ["baxter"] * len(files)
    # iterate through videos
    # compute error for each video
    # if error > threshold, save the video
    dataset = RobotDataset(files, file_labels, config)
    train_loader = DataLoader(
        dataset,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    model = CopyModel()
    window = config.n_past + config.n_future
    T = 31
    pbar = tqdm(initial=0, total=len(dataset) * (floor(T/window)), desc="processing videos")
    num_high_err_videos = 0
    num_high_err_snippets = 0
    num_snippets = 0

    num_gifs = 0
    max_gifs = 10

    """
    Meta dictionary that stores information about each sequence.
    Key: video name
    Value: List of dictionaries containing
            - start / end of sequence
            - cost of sequence
    """
    meta_dict = defaultdict(list)

    all_err = []
    for d in train_loader:
        x = d["images"].transpose_(0, 1) # T x B
        masks = d["masks"].transpose_(0,1) # T x B
        names = d["file_name"] # B
        T = x.shape[0]
        all_losses = defaultdict(float)
        high_err_video = False
        for i in range(floor(T / window)):
            video_name = names[0]
            s = i * window
            e = (i + 1) * window
            batch = x[s:e], masks[s:e], names
            losses = eval_step(batch, model, config)
            num_snippets += 1
            if losses["world_mse"] > 0.01:
                high_err_video = True
                num_high_err_snippets += 1
            entry = {"start": s, "end": e, "world_mse": losses["world_mse"], "high_error": losses["world_mse"] > 0.01}
            meta_dict[video_name].append(entry)
            # if losses["world_mse"] > 0.01 and num_gifs < max_gifs:
            #     # save video
            #     mask = batch[1]
            #     robot_mask = mask.type(torch.bool)
            #     robot_mask = robot_mask.repeat(1,1,3,1,1)
            #     batch[0][robot_mask] *= 0
            #     imgs = (255 * batch[0].squeeze().permute(0,2,3,1)).cpu().numpy().astype(np.uint8)
            #     name = names[0].split("/")[-1][:-5]
            #     err = losses["world_mse"]
            #     gif_name = f"{name}_{losses['world_mse']:.6f}.gif"
            #     imageio.mimwrite(gif_name, imgs)
            #     num_gifs += 1
            all_err.append(losses["world_mse"])
            pbar.update()

        if high_err_video:
            num_high_err_videos += 1
    print(f"number of videos with high errors {num_high_err_videos}/{len(dataset)}")
    print(f"number of snippets with high errror: {num_high_err_snippets}/{num_snippets}")

    # histogram of world err
    # all_err = np.asarray(all_err)
    # print(all_err.mean())
    # plt.hist(all_err, bins=24, color='c', edgecolor='k', alpha=0.65)
    # plt.axvline(all_err.mean(), color='k', linestyle='dashed', linewidth=1)
    # # plt.show()
    # plt.savefig("world_error_histogram.png")

    # save meta dict
    with open(f"baxter_{arm}.pkl", "wb") as f:
        pickle.dump(meta_dict, f)