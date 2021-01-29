import io
import os
import random

from dataclasses import dataclass

from torch.utils.data.dataloader import DataLoader
from typing import Any
import cv2
import h5py
import imageio
import ipdb
import numpy as np
from ipdb import set_trace as st
import pandas as pd
import torch
import torchvision.transforms as tf
import torch.utils.data as data
from src.utils.dataloader import inf_loop_dataloader
from tqdm import trange


class RobotDataset(data.Dataset):
    def __init__(self, robot_name, metadata, hdf5_list, config) -> None:
        self.robot_name = robot_name
        self._traj_names = hdf5_list
        self._config = config
        self._metadata = metadata
        self._data_root = config.data_root
        self._video_length = config.video_length
        self._action_dim = config.action_dim
        self._impute_autograsp_action = config.impute_autograsp_action
        self._img_transform = tf.Compose(
            [tf.ToTensor(), tf.CenterCrop(config.image_width)]
        )
        self._rng = random.Random(config.seed)
        self._memory = {}

    def preload_ram(self):
        # load everything into memory
        for i in trange(len(self._traj_names), f"loading {self.robot_name} into RAM"):
            self._memory[i] = self.__getitem__(i)

    def __getitem__(self, idx):
        """
        Opens the hdf5 file and extracts the videos, actions, states, and masks
        """
        cam_i = 0  # TODO: hyperparameter
        img_dims = [64, 85]  # TODO: hyperparameter
        robonet_root = self._data_root
        name = self._traj_names[idx]
        if idx in self._memory:
            return self._memory[idx]

        hdf5_path = os.path.join(robonet_root, name)
        with h5py.File(hdf5_path, "r") as hf:
            file_metadata = self._metadata.loc[name]
            ep_len = file_metadata["img_T"]
            assert ep_len >= self._video_length, f"{ep_len}, {hdf5_path}"
            start = 0
            end = ep_len
            if ep_len > self._video_length:
                offset = ep_len - self._video_length
                start = np.random.randint(0, offset + 1)
                end = start + self._video_length

            images = self._load_camera_imgs(
                cam_i, hf, file_metadata, img_dims, start, self._video_length
            )
            states = hf["env"]["state"][start:end].astype(np.float32)
            actions = self._load_actions(hf, file_metadata, start, end - 1)
            masks = self._load_masks(hf, cam_i, img_dims, start, end)

            assert (
                len(images) == len(states) == len(actions) + 1 == len(masks)
            ), f"{hdf5_path}, {images.shape}, {states.shape}, {actions.shape}, {masks.shape}"

            # preprocessing
            images = self._preprocess_images(images)
            masks = self._preprocess_masks(masks)
            low = hf["env"]["low_bound"][0]
            high = hf["env"]["high_bound"][0]
            states = self._preprocess_states(states, low, high)
            actions = self._preprocess_actions(actions)

        return images, states, actions, masks

    def _load_camera_imgs(
        self,
        cam_index,
        file_pointer,
        file_metadata,
        target_dims,
        start_time=0,
        n_load=None,
    ):
        cam_group = file_pointer["env"][f"cam{cam_index}_video"]
        old_dims = file_metadata["frame_dim"]
        length = file_metadata["img_T"]
        encoding = file_metadata["img_encoding"]
        image_format = file_metadata["image_format"]

        if n_load is None:
            n_load = length

        old_height, old_width = old_dims
        target_height, target_width = target_dims
        resize_method = cv2.INTER_CUBIC
        if target_height * target_width < old_height * old_width:
            resize_method = cv2.INTER_AREA

        images = np.zeros((n_load, target_height, target_width, 3), dtype=np.uint8)
        if encoding == "mp4":
            buf = io.BytesIO(cam_group["frames"][:].tobytes())
            img_buffer = [
                img
                for t, img in enumerate(imageio.get_reader(buf, format="mp4"))
                if start_time <= t < n_load + start_time
            ]
        elif encoding == "jpg":
            img_buffer = [
                cv2.imdecode(cam_group[f"frame{t}"][:], cv2.IMREAD_COLOR)[:, :, ::-1]
                for t in range(start_time, start_time + n_load)
            ]
        else:
            raise ValueError("encoding not supported")

        for t, img in enumerate(img_buffer):
            if (old_height, old_width) == (target_height, target_width):
                images[t] = img
            else:
                images[t] = cv2.resize(
                    img, (target_width, target_height), interpolation=resize_method
                )

        if image_format == "RGB":
            return images
        elif image_format == "BGR":
            return images[:, :, :, ::-1]
        raise NotImplementedError

    def _load_masks(self, file_pointer, cam_index, target_dims, start, end):
        masks = file_pointer["env"][f"cam{cam_index}_mask"][start:end]
        old_dims = masks.shape[-2:]
        old_height, old_width = old_dims
        target_height, target_width = target_dims
        resize_method = cv2.INTER_CUBIC
        if target_height * target_width < old_height * old_width:
            resize_method = cv2.INTER_AREA
        if (old_height, old_width) == (target_height, target_width):
            return masks

        resized_masks = []
        for img in masks:
            rs_mask = (
                cv2.resize(
                    255.0 * img,
                    (target_width, target_height),
                    interpolation=resize_method,
                )
                / 255.0
            ).astype(np.float32)
            resized_masks.append(rs_mask)
        return resized_masks

    def _load_actions(self, file_pointer, meta_data, start, end):
        a_T, adim = meta_data["action_T"], meta_data["adim"]
        if self._action_dim == adim:
            return file_pointer["env"]["actions"][start:end].astype(np.float32)
        elif (
            self._impute_autograsp_action
            and adim + 1 == self._action_dim
            and meta_data["primitives"] == "autograsp"
        ):
            action_append, old_actions = (
                np.zeros((a_T, 1)),
                file_pointer["policy"]["actions"][:],
            )
            next_state = file_pointer["env"]["state"][:][1:, -1]
            high_val, low_val = meta_data["high_bound"][-1], meta_data["low_bound"][-1]
            midpoint = (high_val + low_val) / 2.0

            for t, s in enumerate(next_state):
                if s > midpoint:
                    action_append[t, 0] = high_val
                else:
                    action_append[t, 0] = low_val
            new_actions = np.concatenate((old_actions, action_append), axis=-1)
            return new_actions[start:end].astype(np.float32)
        else:
            raise ValueError(f"file adim {adim}, target adim {self._action_dim}, primitive {meta_data['primitives']}")

    def _preprocess_images(self, images):
        """
        Converts numpy image (uint8) [0, 255] to tensor (float) [0, 1].
        """
        video_tensor = torch.stack([self._img_transform(i) for i in images])
        return video_tensor

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

    def _preprocess_actions(self, actions):
        return torch.from_numpy(actions)

    def _preprocess_masks(self, masks):
        mask_tensor = torch.stack([self._img_transform(i) for i in masks])
        return mask_tensor

    def __len__(self):
        return len(self._traj_names)


class WidowXDataset(RobotDataset):
    def __init__(self, metadata, config, hdf5_list=None) -> None:
        """
        metadata: robonet's metadata dataframe
        config: argparser namespace
        hdf5_list: if passed, directly instantiate with this list
        """
        if hdf5_list is None:
            widowx_df = metadata.loc["widowx" == metadata["robot"]]
            widowx_subset = widowx_df[widowx_df["camera_configuration"] == "widowx1"]
            hdf5_list = widowx_subset.index.tolist()
        super().__init__("widowx1", metadata, hdf5_list, config)

    def _preprocess_states(self, states, low, high):
        # TODO: add offset to move the end effector position to the tip
        states = torch.from_numpy(states)
        force_min, force_max = low[4], high[4]
        states[:, 4] = (states[:, 4] - force_min) / (force_max - force_min)
        return states


class BaxterDataset(RobotDataset):
    def __init__(self, metadata, config, hdf5_list=None) -> None:
        """
        metadata: robonet's metadata dataframe
        config: argparser namespace
        """
        if hdf5_list is None:
            baxter_df = metadata.loc["baxter" == metadata["robot"]]
            hdf5_list = baxter_df.index.tolist()
        super().__init__("baxter", metadata, hdf5_list, config)


class SawyerDataset(RobotDataset):
    def __init__(self, metadata, config, hdf5_list=None) -> None:
        """
        metadata: robonet's metadata dataframe
        config: argparser namespace
        """
        if hdf5_list is None:
            sawyer_df = metadata.loc["sawyer" == metadata["robot"]]
            sawyer_subset = sawyer_df["sudri0" == sawyer_df["camera_configuration"]]
            hdf5_list = sawyer_subset.index
        super().__init__("sawyer_sudri0", metadata, hdf5_list, config)

    def _preprocess_states(self, states, low, high):
        # TODO: add offset to move the end effector position to the tip
        # perhaps we don't want to because sawyer has multiple grippers, so wrist makes more sense.
        states = torch.from_numpy(states)
        force_min, force_max = low[4], high[4]
        states[:, 4] = (states[:, 4] - force_min) / (force_max - force_min)
        return states


class MultiRobotDataset:
    """
    Combines multiple RobotDataset into one for cycling over all robot datasets
    robot_datasets: a list of datasets per robot dataset
    """

    def __init__(self, config, robot_datasets=None) -> None:
        self._config = config
        self._batch_size = config.batch_size
        if robot_datasets is None:
            self._robots = self._create_robot_dataloaders(config)
        else:
            self._robots = self._init_dataloaders(robot_datasets)
        self._num_robots = len(self._robots)
        assert self._batch_size % self._num_robots == 0, "multirobot batch size must be multiple of |robots|"

    def _create_robot_dataloaders(self, config):
        """
        Initializes the robot dataloaders
        """
        metadata_path = os.path.join(config.data_root, "meta_data.pkl")
        metadata = pd.read_pickle(metadata_path, compression="gzip")
        # TODO: devise some automatic intialization from cli args
        widowx = WidowXDataset(metadata, config)
        baxter = BaxterDataset(metadata, config)
        sawyer = SawyerDataset(metadata, config)
        datasets = [widowx, baxter, sawyer]
        loaders = self._init_dataloaders(datasets)
        return loaders

    def _init_dataloaders(self, datasets):
        loaders = []
        self._robot_names = []
        for d in datasets:
            self._robot_names.append(d.robot_name)
            l = DataLoader(d,
                num_workers=self._config.data_threads,
                batch_size=self._batch_size // len(datasets),
                shuffle=True,
                drop_last=True,
                pin_memory=True
            )
            loaders.append(inf_loop_dataloader(l))
        return loaders

    def __iter__(self):
        generator = self._get_video()
        while True:
            yield next(generator)



    def _make_batch(self):
        """
        Collects robot videos into a minibatch
        """
        generator = self._get_video()
        # collate the images, masks, states, actions by batch dim
        images = []
        states = []
        actions = []
        masks = []
        # robot_names = []
        # file_names = []
        # TODO: parallelize this part using Ray or something.
        for _ in range(self._num_robots):
            sample = next(generator)
            i, s, a, m = sample
            images.append(i)
            states.append(s)
            actions.append(a)
            masks.append(m)
            # robot_names.append(sample.robot_name)
            # file_names.append(sample.file_name)
        # B L C W H dim
        images = torch.cat(images)
        states = torch.cat(states)
        actions = torch.cat(actions)
        masks = torch.cat(masks)
        # batch_sample = BatchVideoSample(
        #     images, masks, states, actions, robot_names, file_names
        # )
        # return batch_sample
        return images, states, actions, masks,

    def _get_video(self):
        """
        Cycles through each robot, getting one video per robot
        """
        while True:
            for i, robot_dataloader in enumerate(self._robots):
                self.current_robot = self._robot_names[i]
                yield next(robot_dataloader)

if __name__ == "__main__":
    from src.config import argparser

    config, _ = argparser()
    config.data_root = "/home/ed/Robonet/hdf5"
    config.batch_size = 3  # needs to be multiple of the # of robots
    config.video_length = 31
    config.image_width = 64
    config.impute_autograsp_action = True
    config.action_dim = 5
    data = MultiRobotDataset(config)

    for i, batch in enumerate(data):
        imgs, states, actions, masks = batch
        for robot_imgs, robot_masks in zip(imgs, masks):
            # B x C x H x W
            # B x H x W x C
            img_gif = robot_imgs.permute(0, 2, 3, 1).clamp_(0, 1).cpu().numpy()
            img_gif = np.uint8(img_gif * 255)
            robot_masks = robot_masks.cpu().numpy().squeeze().astype(bool)
            img_gif[robot_masks] = (0, 255, 255)
            imageio.mimwrite(f"test{i}.gif", img_gif)
            break

        print()
        if i >= 8:
            break