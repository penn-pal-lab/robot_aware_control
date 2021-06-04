import os

import h5py
import numpy as np
import pickle
import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as F
import torch.utils.data as data
from tqdm import trange
from src.utils.gaussian import gaus2d
from src.utils.camera_calibration import (
    world_to_camera_dict,
    cam_intrinsics_dict,
    camera_to_world_dict,
)
import ipdb


class RoboNetDataset(data.Dataset):
    def __init__(
        self, hdf5_list, robot_list, config, augment_img=False, load_snippet=False
    ):
        """
        hdf5_list: list of hdf5 files to load
        robot_list: list of robot_viewpoint names
        split: load a random snippet, or the entire video depending on split.
        """
        self._traj_names = hdf5_list
        self._traj_robots = robot_list
        self._config = config
        self._data_root = config.data_root
        if config.load_movement_info:
            # check for movement info dictionaries
            movement_vp = {}
            for name in self._traj_names:
                folder = os.path.basename(os.path.dirname(name))
                if folder in movement_vp:
                    continue

                dict_path = os.path.join(os.path.dirname(name), "obj_movement.pkl")
                with open(dict_path, "rb") as f:
                    info = pickle.load(f)
                movement_vp[folder] = info
            self._movement_vp = movement_vp

        self._video_length = config.video_length
        if load_snippet:
            self._video_length = config.n_past + config.n_future
        self._action_dim = config.action_dim
        self._impute_autograsp_action = config.impute_autograsp_action
        self._augment_img = augment_img
        # assume image is H x W (64, 85), resize into 48 x 64
        w, h = config.image_width, config.image_height
        self._img_transform = tf.Compose([tf.ToTensor(), tf.Resize((h, w))])
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
        Opens the hdf5 file and extracts the videos, actions, states, and masks.

        For robonet data,
        """
        if idx in self._memory:
            return self._memory[idx]

        robonet_root = self._data_root
        name = self._traj_names[idx]
        robot_viewpoint = self._traj_robots[idx]
        hdf5_path = os.path.join(robonet_root, name)
        with h5py.File(hdf5_path, "r") as hf:
            assert "frames" in hf or "observations" in hf
            IMAGE_KEY = "frames"
            if "observations" in hf:
                IMAGE_KEY = "observations"

            MASK_KEY = "mask"
            if "masks" in hf:
                MASK_KEY = "masks"

            ep_len = hf[IMAGE_KEY].shape[0]
            assert ep_len >= self._video_length, f"{ep_len}, {hdf5_path}"
            start = 0
            end = ep_len
            if ep_len > self._video_length:
                offset = ep_len - self._video_length
                start = self._rng.randint(0, offset + 1)
                end = start + self._video_length

            images = hf[IMAGE_KEY][start:end]
            raw_low, raw_high = self._load_bounds(hf, robot_viewpoint, idx)
            g_low, g_high = raw_low[4], raw_high[4]
            # For robonet data, states is normalized. For locobot, states is raw.
            states = self._load_states(hf, start, end)
            actions = self._load_actions(hf, g_low, g_high, start, end - 1)
            if self._config.preprocess_action != "raw":
                raw_states = states.copy()
                raw_actions = actions.copy()
            masks = hf[MASK_KEY][start:end].astype(np.float32)
            qpos = self._load_qpos(hf, start, end)
            assert (
                len(images) == len(states) == len(actions) + 1 == len(masks)
            ), f"{hdf5_path}, {images.shape}, {states.shape}, {actions.shape}, {masks.shape}"

            # preprocessing
            low, high = self._preprocess_bounds(raw_low, raw_high, idx)
            images, masks = self._preprocess_images_masks(images, masks)
            # normalize and transform the states
            states = self._preprocess_states(states, low, high, robot_viewpoint, idx)
            # create the actions
            actions = self._preprocess_actions(states, actions, low, high, idx)
            robot = "locobot" if "locobot" in robot_viewpoint else hf.attrs["robot"]
            folder = os.path.basename(os.path.dirname(name))
            if self._config.model_use_heatmap:
                # heatmaps = create_heatmaps(states, low, high, robot, folder)
                # need to account for camera space state
                raise NotImplementedError
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
        if self._config.model_use_heatmap:
            # out["heatmaps"] = heatmaps
            raise NotImplementedError
        # load info necessary for analytical model output
        if "finetune" in self._config.experiment:
            # needed to denormalize states for state / heatmap prediction
            out["low"] = low
            out["high"] = high
            if self._config.preprocess_action == "raw":
                pass
            elif "camera" in self._config.preprocess_action:
                # needed for normalizing / denormalizing raw states
                out["raw_low"] = raw_low
                out["raw_high"] = raw_high
                out["raw_actions"] = raw_actions
                # locobot analytical model assumes states are normalized in world frame
                raw_states[:, :3] = normalize(raw_states[:, :3], raw_low[:3], raw_high[:3])
                raw_states[:, 4] = normalize(raw_states[:, 4], raw_low[4], raw_high[4])
                out["raw_states"] = raw_states
            else:
                raise NotImplementedError

        if self._config.load_movement_info:
            out["high_movement"] = self._movement_vp[folder][hdf5_path]
        return out

    def _load_actions(self, file_pointer, gripper_low, gripper_high, start, end):
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
            high_val, low_val = gripper_high[-1], gripper_low[-1]
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

    def _load_bounds(self, file_pointer, robot_viewpoint, idx):
        """
        Load the bounds of the workspace. If in camera space, determine the new bounds.
        """
        if "locobot" in robot_viewpoint:
            low = np.array([0.015, -0.3, 0.1, 0, 0], dtype=np.float32)
            high = np.array([0.55, 0.3, 0.4, 1, 1], dtype=np.float32)
        else:
            low = file_pointer["low_bound"][:]
            high = file_pointer["high_bound"][:]
        return low, high

    def _load_states(self, file_pointer, start, end):
        states = file_pointer["states"][start:end].astype(np.float32)
        if states.shape[-1] != self._config.robot_dim:
            assert self._config.robot_dim > states.shape[-1]
            pad = self._config.robot_dim - states.shape[-1]
            states = np.pad(states, [(0, 0), (0, pad)])
        return states

    def _load_qpos(self, file_pointer, start, end):
        qpos = file_pointer["qpos"][start:end].astype(np.float32)
        if qpos.shape[-1] != self._config.robot_joint_dim:
            assert self._config.robot_joint_dim > qpos.shape[-1]
            pad = self._config.robot_joint_dim - qpos.shape[-1]
            qpos = np.pad(qpos, [(0, 0), (0, pad)])
        return qpos

    def _preprocess_bounds(self, low, high, idx):
        low, high = low.copy(), high.copy()
        if "camera" in self._config.preprocess_action:
            # project bounding box into camera space
            x_min, x_max = low[0], high[0]
            y_min, y_max = low[1], high[1]
            z_min, z_max = low[2], high[2]
            bounding_box = np.array(
                [
                    [x_min, y_min, z_min],
                    [x_min, y_min, z_max],
                    [x_min, y_max, z_min],
                    [x_min, y_max, z_max],
                    [x_max, y_min, z_min],
                    [x_max, y_min, z_max],
                    [x_max, y_max, z_min],
                    [x_max, y_max, z_max],
                ]
            )  # (8,3)
            robot_type = self._traj_robots[idx]
            world2cam = world_to_camera_dict[robot_type]  # (4,4)
            # make coordinate [x,y,z,1]
            bounding_box = np.concatenate(
                [bounding_box, np.ones((bounding_box.shape[0], 1))], 1
            ).T
            # transform bounding box to camera frame
            c_bounding_box = ((world2cam @ bounding_box).T)[:, :3]  # (8, 3)
            # get new min and max on each axis
            low[:3] = np.min(c_bounding_box, 0)
            high[:3] = np.max(c_bounding_box, 0)
        return low, high

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

    def _preprocess_states(self, states, low, high, robot_viewpoint, idx):
        """
        states: T x 5 array of [x,y,z, r, f], r is rotation in radians, f is gripper force
        States may or may not be normalized in xyz axis, depending on robonet or locobot
        Normalize x,y,z, f dimensions
        yaw is in radians across all robots, so it is consistent.
        """
        states = states.copy()
        # first get raw states
        if "locobot" in robot_viewpoint:
            eef_pos = states[:, :3]
        else:
            eef_pos = denormalize(states[:, :3], low[:3], high[:3])

        # transform into camera frame if necessary
        if "camera" in self._config.preprocess_action:
            robot_type = self._traj_robots[idx]
            world2cam = world_to_camera_dict[robot_type]  # (4,4)
            # make coordinate [x,y,z,1]
            eef_pos = np.concatenate([eef_pos, np.ones((eef_pos.shape[0], 1))], 1).T
            # transform bounding box to camera frame
            eef_pos = ((world2cam @ eef_pos).T)[:, :3]  # (T,3)

        # normalize xyz dims of states
        states[:, :3] = normalize(eef_pos, low[:3], high[:3])
        # normalize gripper force
        states[:, 4] = normalize(states[:, 4] , low[4], high[4])
        return states

    def _preprocess_actions(self, states, actions, low, high, idx):
        strategy = self._config.preprocess_action
        if strategy == "raw":
            return torch.from_numpy(actions)
        elif strategy == "state_infer":
            # states = states.copy()
            # self._impute_true_actions(states, actions, low, high)
            # return torch.from_numpy(actions)
            raise NotImplementedError
        robot_type = self._traj_robots[idx]
        world2cam = world_to_camera_dict[robot_type]
        states = states.copy()
        if strategy == "camera_raw":
            cam2world = camera_to_world_dict[robot_type]
            actions = self._make_camera_actions(
                states, actions, world2cam, cam2world, low, high
            )
        elif strategy == "camera_state_infer":
            # self._impute_camera_actions(states, actions, world2cam, low, high)
            raise NotImplementedError
        return torch.from_numpy(actions)

    def _convert_world_to_camera_pos(self, state, w_to_c):
        e_to_w = np.eye(4)
        e_to_w[:3, 3] = state[:3]
        e_to_c = w_to_c @ e_to_w
        pos_c = e_to_c[:3, 3]
        return pos_c

    def _make_camera_actions(self, states, actions, w_to_c, c_to_w, low, high):
        """
        Concert raw actions to camera frame displacements

        States is already in normalized, camera frame
        Actions is in raw world frame, end effector displacements

        To convert displacements to another frame, we first project s and (s+a) to camera frame, and then take the difference.
        We don't convert the rotation dimension for now.
        """
        actions = actions.copy()
        actions = np.zeros_like(actions)
        # convert normalized camera frame state back to raw world frame
        c_eef_pos = denormalize(states[:, :3], low[:3], high[:3])
        c_eef_pos = np.concatenate([c_eef_pos, np.ones((c_eef_pos.shape[0], 1))], 1).T
        eef_pos = ((c_to_w @ c_eef_pos).T)[:-1, :3]  # (T-1,3)
        next_eef_pos = eef_pos + actions[:, :3]  # (T-1, 3)
        # convert unnormalized world coords to camera frame
        eef_pos = np.concatenate([eef_pos, np.ones((eef_pos.shape[0], 1))], 1).T
        c_eef_pos = ((w_to_c @ eef_pos).T)[:, :3]
        next_eef_pos = np.concatenate(
            [next_eef_pos, np.ones((next_eef_pos.shape[0], 1))], 1
        ).T
        c_next_eef_pos = ((w_to_c @ next_eef_pos).T)[:, :3]
        actions[:, :3] = c_next_eef_pos - c_eef_pos
        return actions

    def _impute_camera_actions(self, states, actions, w_to_c, low, high):
        """
        Just calculate the true offset between states instead of using  the recorded actions.
        """
        states[:, :3] = denormalize(states[:, :3], low[:3], high[:3])
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
        states[:, :3] = denormalize(states[:, :3], low[:3], high[:3])
        for t in range(len(actions)):
            state = states[t][:3]
            next_state = states[t + 1][:3]
            true_offset_c = next_state - state
            actions[t][:3] = true_offset_c

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
    """Changes tensor idx from batch-first to time-first"""
    transpose_keys = [
        "qpos",
        "images",
        "states",
        "actions",
        "masks",
        "heatmaps",
        "raw_actions",
        "raw_states",
    ]
    meta_keys = ["robot", "folder", "file_path", "idx"]
    # transpose from (B, L, C, W, H) to (L, B, C, W, H)
    for k in transpose_keys:
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


def denormalize(states, low, high):
    states = states * (high - low)
    states = states + low
    return states


def normalize(states, low, high):
    states = states - low
    states = states / (high - low)
    return states


def create_heatmaps(states, low, high, robot, viewpoint):
    """Create eef heatmaps of the robot

    Args:
        states (numpy array): normalized states
        low (numpy array): lower bound
        high (numpy array): upper bound
        robot (str): type of robot
        viewpoint (str): robot viewpoint
    """
    # first denormalize the eef states
    states = states.clone()
    states[:, :3] = denormalize(states[:, :3], low[:3], high[:3])
    eef_pos = states[:, :3].numpy()
    # depending on the robot configuration, add Z offset to the gripper
    if robot == "sawyer":
        eef_pos[:, 2] -= 0.15
        wTc = world_to_camera_dict[f"sawyer_{viewpoint}"]
        cam_intrinsics = cam_intrinsics_dict["logitech_c420"]
        odim = (320, 240)
    elif robot == "baxter":
        # TODO: account for baxter arm, and viewpoint
        wTc = world_to_camera_dict[f"baxter_{viewpoint}"]
        cam_intrinsics = cam_intrinsics_dict["logitech_c420"]
        odim = (320, 240)
    elif robot == "widowx":
        # since widowx is mounted upside down, we add positive z
        eef_pos[:, 2] += 0.05
        wTc = world_to_camera_dict[f"widowx_{viewpoint}"]
        cam_intrinsics = cam_intrinsics_dict["logitech_c420"]
        odim = (320, 240)
    elif robot == "locobot":
        wTc = world_to_camera_dict[f"locobot_c0"]
        cam_intrinsics = cam_intrinsics_dict["intel_realsense_d435"]
        odim = (640, 480)
    else:
        raise ValueError

    eef_pos = np.concatenate([eef_pos, np.ones((eef_pos.shape[0], 1))], 1).T
    w, h = tdim = (64, 48)  # horizontal, vertical dim
    all_pix_2d = get_2d_eef_pos(eef_pos, cam_intrinsics, wTc, tdim, odim)
    # only get 2d coordinates within image dims
    valid_timesteps = ((0 <= all_pix_2d[0]) & (all_pix_2d[0] < w)) & (
        (0 <= all_pix_2d[1]) & (all_pix_2d[1] < h)
    )

    # define normalized 2D gaussian
    x = np.arange(0, w)
    y = np.arange(0, h)
    x, y = np.meshgrid(x, y)  # get 2D variables instead of 1D
    heatmaps = []
    for i in range(all_pix_2d.shape[1]):
        if valid_timesteps[i]:
            z = gaus2d(
                x, y, mx=all_pix_2d[0, i], my=all_pix_2d[1, i], sx=5, sy=5, height=100
            )
            z = np.clip(z, 0, 1)
        else:
            z = np.zeros((h, w))
        heatmaps.append(z)
    heatmaps = np.asarray(heatmaps)
    heatmaps = np.expand_dims(heatmaps, 1).astype(np.float32)
    return heatmaps


if __name__ == "__main__":
    from src.utils.plot import save_gif
    from src.config import argparser

    config, _ = argparser()
    config.data_root = "/scratch/anonymous/Robonet"
    config.batch_size = 16  # needs to be multiple of the # of robots
    config.video_length = 31
    config.image_width = 64
    # config.impute_autograsp_action = True
    config.data_threads = 0
    config.action_dim = 5
    config.model_use_heatmap = True

    # from src.dataset.sawyer_multiview_dataloaders import create_loaders
    # from src.dataset.finetune_multirobot_dataloaders import create_transfer_loader
    # from src.dataset.finetune_widowx_dataloaders import create_transfer_loader
    # loader = create_transfer_loader(config)

    # for data in loader:
    #     images = data["images"]
    #     # states = data["states"]
    #     heatmaps = data["heatmaps"].repeat(1,1,3,1,1)
    #     heat_images = (images * heatmaps).transpose(0,1).unsqueeze(2)
    #     original_images = images.transpose(0,1).unsqueeze(2)
    #     gif = torch.cat([original_images, heat_images], 2)
    #     save_gif("batch.gif", gif)
    #     break
    #     # apply heatmap to images
    #     # eef_images = ((255 * heatmaps[0] * images[0]).permute(0,2,3,1).numpy().astype(np.uint8))
    #     # imageio.mimwrite("eef.gif", eef_images)