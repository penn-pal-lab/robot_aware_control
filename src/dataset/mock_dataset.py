from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as tf


class VideoDataset(data.Dataset):
    def __init__(self, files, config):
        self._files = files
        # shift [0,1] to [-1,1]
        self._img_transforms = tf.Compose([tf.Lambda(seq_to_tensor)])
        self._action_dim = config.action_dim
        self._cache = LRUCache(100000)
        self._cf = config
        self._horizon = config.n_past + config.n_future

    def __getitem__(self, index):
        path = self._files[index]
        item = self._cache.get(path)
        if item is not None:
            return item
        with h5py.File(path, "r") as hf:
            # first check how long video is
            ep_len = hf["frames"].shape[0]
            assert ep_len >= self._horizon, f"{ep_len}, {path}"
            # if video is longer than horizon, sample a starting point
            start = 0
            end = self._horizon
            frames_shape = list(hf["frames"].shape)
            robot_shape = list(hf["robot"].shape)
            actions_shape = list(hf["actions"].shape)
            if ep_len > self._horizon:
                offset = ep_len - self._horizon
                start = np.random.randint(0, offset + 1)
                end = start + self._horizon
                frames_shape[0] = robot_shape[0] = self._horizon
                actions_shape[0] = self._horizon - 1

            # frames should be L x C x H x W
            frames = np.zeros(frames_shape, dtype=np.uint8)
            hf["frames"].read_direct(frames, source_sel=np.s_[start:end])
            frames = self._img_transforms(frames)

            robot = np.zeros(robot_shape, dtype=np.float32)
            hf["robot"].read_direct(robot, source_sel=np.s_[start:end])

            actions = np.zeros(actions_shape, dtype=np.float32)
            hf["actions"].read_direct(actions, source_sel=np.s_[start:end - 1])

        # double check dimensions of data
        assert robot.shape[0] == frames.shape[0], f"{robot.shape}, {frames.shape}"
        assert robot.shape[0] - 1 == actions.shape[0], f"{robot.shape}, {actions.shape}"
        assert frames.shape[0] == self._horizon, f"{path}, {frames.shape}"
        assert frames.shape[1] == self._cf.channels, f"{path}, {frames.shape}"
        assert frames.shape[2] == self._cf.image_width, f"{path}, {frames.shape}"
        assert actions.shape[-1] == self._cf.action_dim, f"{path}, {actions.shape}"
        assert robot.shape[-1] == self._cf.robot_dim, f"{path}, {robot.shape}"

        # add to cache
        self._cache.put(path, (frames, robot, actions))
        return frames, robot, actions

    def __len__(self):
        return len(self._files)


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def seq_to_tensor(seq):
    return torch.stack([tf.ToTensor()(i) for i in seq])
