import os

from torch.utils.data.dataloader import DataLoader
from typing import Any
import cv2
import h5py
import imageio
import ipdb
import numpy as np
from ipdb import set_trace as st
import torch
from torchvision.datasets.folder import has_file_allowed_extension
import torchvision.transforms as tf
import torchvision.transforms.functional as F
import torch.utils.data as data
from tqdm import trange
from src.utils.gaussian import gaus2d
from src.utils.camera_calibration import world_to_camera_dict, cam_intrinsics_dict
import ipdb

class RobotDataset(data.Dataset):
    def __init__(
        self, hdf5_list, robot_list, config, augment_img=False, load_snippet=False
    ):
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
        # assume image is H x W (64, 85), resize into 48 x 64
        w, h = config.image_width, config.image_height
        self._img_transform = tf.Compose([tf.ToTensor(), tf.Resize((h, w))])
        # self._img_transform = tf.Compose(
        #     [tf.ToTensor(), tf.CenterCrop(config.image_width)]
        # )

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
            qpos = hf["qpos"][start:end].astype(np.float32)
            # qpos = np.zeros((31, 7))

            assert (
                len(images) == len(states) == len(actions) + 1 == len(masks)
            ), f"{hdf5_path}, {images.shape}, {states.shape}, {actions.shape}, {masks.shape}"

            # preprocessing
            images, masks = self._preprocess_images_masks(images, masks)
            actions = self._preprocess_actions(states, actions, low, high, idx)
            states = self._preprocess_states(states, low, high)
            robot = hf.attrs["robot"]
            folder = os.path.basename(os.path.dirname(name))
            if self._config.load_eef_heatmap:
                heatmaps = self._create_heatmaps(states, low, high, images, robot, folder)
        out = {
            "images": images,
            "states": states,
            "actions": actions,
            "masks": masks,
            "robot": robot,
            "folder": folder,
            "file_path": hdf5_path,
            "idx": idx,
            "qpos": qpos,
        }
        if self._config.load_eef_heatmap:
            out["heatmaps"] = heatmaps
        return out

    def _create_heatmaps(self, states, low, high, images, robot, folder):
         # first denormalize the eef states
        states[:, :3] = self._denormalize(states[:, :3], low[:3], high[:3])
        eef_pos = states[:, :3].numpy()
        # depending on the robot configuration, add Z offset to the gripper
        if robot =="sawyer":
            eef_pos[:, 2] -= 0.15
            wTc = world_to_camera_dict[f"sawyer_{folder}"]
        elif robot == "baxter":
            #TODO: account for baxter arm, and viewpoint
            wTc = world_to_camera_dict[f"baxter_left"]
        elif robot == "widowx":
            # since widowx is mounted upside down, we add positive z
            eef_pos[:, 2] += 0.05
            wTc = world_to_camera_dict[f"widowx1"]
        else:
            raise ValueError

        eef_pos = np.concatenate([eef_pos, np.ones((eef_pos.shape[0], 1))], 1).T
        # assumes image is (240, 320)
        # assumes image was captured from robonet, logitech 420 camera intrinsics
        cam_intrinsics = cam_intrinsics_dict["logitech_c420"]
        # projM = cam_intrinsics @ wTc[:3]
        # all_pix_3d = projM @ eef_pos
        # all_pix_3d /= all_pix_3d[2]
        # all_pix_2d = all_pix_3d[:2]
        # all_pix_2d[0] *= 64 / 320
        # all_pix_2d[1] *= 48 / 240

        tdim = (64, 48) # horizontal, vertical dim
        odim = (320, 240)
        all_pix_2d = get_2d_eef_pos(eef_pos, cam_intrinsics, wTc, tdim, odim)
        # only get 2d coordinates in image range
        valid_timesteps = ((0 <= all_pix_2d[0]) & (all_pix_2d[0] < 64))& ((0 <= all_pix_2d[1]) & (all_pix_2d[1]< 48))

        # define normalized 2D gaussian
        x = np.arange(0, 64)
        y = np.arange(0, 48)
        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
        heatmaps = []
        for i in range(all_pix_2d.shape[1]):
            if valid_timesteps[i]:
                z = gaus2d(x, y, mx=all_pix_2d[0, i], my=all_pix_2d[1, i], sx=5, sy=5,height=100)
                z = np.clip(z, 0, 1)
            else:
                z = np.zeros((48, 64))
            heatmaps.append(z)
        heatmaps = np.asarray(heatmaps)
        heatmaps = np.expand_dims(heatmaps, 1)
        return heatmaps

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
            mask_tensor = (
                torch.stack([self._img_transform(i) for i in masks])
                .type(torch.bool)
                .type(torch.float32)
            )
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
            world2cam = world_to_camera_dict["sawyer_sudri0_c0"]
        elif self._config.training_regime == "train_sawyer_multiview":
            world2cam = world_to_camera_dict["sawyer_" + robot_type]
        elif self._config.training_regime == "finetune":
            # finetune on baxter, convert to camera frame
            # Assumes the baxter arm is right arm!
            arm = "left" if "left" in filename else "right"
            world2cam = world_to_camera_dict[f"baxter_{arm}"]
        else:
            raise ValueError
        states = states.copy()
        if strategy == "camera_raw":
            self._convert_actions(states, actions, world2cam, low, high)
        elif strategy == "camera_state_infer":
            self._impute_camera_actions(states, actions, world2cam, low, high)
        return torch.from_numpy(actions)

    def __len__(self):
        return len(self._traj_names)


def get_2d_eef_pos(state, cam_intrinsics, world_to_cam, target_dim, orig_dim):
    # assumes image is (240, 320)
    projM = cam_intrinsics @ world_to_cam[:3]
    all_pix_3d = projM @ state
    all_pix_3d /= all_pix_3d[2]
    all_pix_2d = all_pix_3d[:2]
    all_pix_2d[0] *= target_dim[0] / orig_dim[0]
    all_pix_2d[1] *= target_dim[1] / orig_dim[1]

    # only get 2d coordinates in image range
    all_pix_2d = all_pix_2d.astype(np.uint8)
    return all_pix_2d

def process_batch(data, device):
    data_keys = ["qpos", "images", "states", "actions", "masks", "heatmaps"]
    meta_keys = ["robot", "folder", "file_path", "idx"]
    # transpose from (B, L, C, W, H) to (L, B, C, W, H)
    for k in data_keys:
        if k in data:
            data[k] = data[k].transpose_(1, 0).to(device, non_blocking=True)
    return data


def get_batch(loader, device):
    """Infinite batch generator for dataloader

    Args:
        loader (Dataloader): dataloader to get batches from
        device (torch.Device): GPU to put tensors on

    Yields:
        [type]: [description]
    """

    while True:
        for data in loader:
            yield process_batch(data, device)


if __name__ == "__main__":
    from src.utils.plot import save_gif
    from src.config import argparser
    from src.utils.camera_calibration import camera_to_world_dict

    config, _ = argparser()
    config.data_root = "/scratch/edward/Robonet"
    config.batch_size = 16  # needs to be multiple of the # of robots
    config.video_length = 31
    config.image_width = 64
    # config.impute_autograsp_action = True
    config.data_threads = 0
    config.action_dim = 5
    config.load_eef_heatmap = True

    # from src.dataset.sawyer_multiview_dataloaders import create_loaders
    from src.dataset.finetune_multirobot_dataloaders import create_transfer_loader
    # from src.dataset.finetune_widowx_dataloaders import create_transfer_loader
    loader = create_transfer_loader(config)

    for data in loader:
        images = data["images"]
        # states = data["states"]
        heatmaps = data["heatmaps"].repeat(1,1,3,1,1)
        heat_images = (images * heatmaps).transpose(0,1).unsqueeze(2)
        original_images = images.transpose(0,1).unsqueeze(2)
        gif = torch.cat([original_images, heat_images], 2)
        save_gif("batch.gif", gif)
        break
        # apply heatmap to images
        # eef_images = ((255 * heatmaps[0] * images[0]).permute(0,2,3,1).numpy().astype(np.uint8))
        # imageio.mimwrite("eef.gif", eef_images)