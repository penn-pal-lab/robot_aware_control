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
        self._cache = LRUCache(1000)
        self._cf = config

    def __getitem__(self, index):
        path = self._files[index]
        item = self._cache.get(path)
        if item is not None:
            return item
        with h5py.File(path, "r") as hf:
            frames = np.zeros(hf["frames"].shape, dtype=np.uint8)
            hf["frames"].read_direct(frames)
            # frames should be L x C x H x W
            frames = self._img_transforms(frames)
            robot = np.asarray(hf["robot"][:], dtype=np.float32)
            actions = np.asarray(hf["actions"][:], dtype=np.float32)

        # double check dimensions of data
        assert robot.shape[0] == frames.shape[0], f"{robot.shape}, {frames.shape}"
        assert robot.shape[0] - 1 == actions.shape[0], f"{robot.shape}, {actions.shape}"
        assert (
            frames.shape[0] == self._cf.n_past + self._cf.n_future
        ), f"{path}, {frames.shape}"
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