from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as tf
from tqdm import tqdm


class VideoDataset(data.Dataset):
    def __init__(self, files, config):
        self._files = files
        self._img_transforms = tf.Compose([tf.Lambda(seq_to_tensor)])
        self._action_dim = config.action_dim
        # self._cache = LRUCache(100000)
        self._cf = config
        self._horizon = config.n_past + config.n_future
        self._data = []

        # try loading everything
        for path in tqdm(files, desc="loading data into ram"):
            with h5py.File(path, "r") as hf:
                # first check how long video is
                ep_len = hf["object_inpaint_demo"].shape[0]
                assert ep_len >= self._horizon, f"{ep_len}, {path}"
                # if video is longer than horizon, sample a starting point
                start = 0
                frames_shape = list(hf["object_inpaint_demo"].shape)
                robot_shape = list(hf["robot_state"].shape)
                actions_shape = list(hf["actions"].shape)
                masks_shape = list(hf["masks"].shape)

                # frames should be L x C x H x W
                frames = np.zeros(frames_shape, dtype=np.uint8)
                hf["object_inpaint_demo"].read_direct(frames)
                frames = self._img_transforms(frames)

                robot = np.zeros(robot_shape, dtype=np.float32)
                hf["robot_state"].read_direct(robot)

                actions = np.zeros(actions_shape, dtype=np.float32)
                hf["actions"].read_direct(actions)

                masks = np.zeros(masks_shape, dtype=np.bool)
                hf["masks"].read_direct(masks)

                self._data.append((frames, robot, actions, masks))

    def __getitem__(self, index):
        path = self._files[index]
        # item = self._cache.get(path)
        # if item is None:
        #     with h5py.File(path, "r") as hf:
        #         # first check how long video is
        #         ep_len = hf["frames"].shape[0]
        #         assert ep_len >= self._horizon, f"{ep_len}, {path}"
        #         # if video is longer than horizon, sample a starting point
        #         start = 0
        #         frames_shape = list(hf["frames"].shape)
        #         robot_shape = list(hf["robot"].shape)
        #         actions_shape = list(hf["actions"].shape)

        #         # frames should be L x C x H x W
        #         frames = np.zeros(frames_shape, dtype=np.uint8)
        #         hf["frames"].read_direct(frames)
        #         frames = self._img_transforms(frames)

        #         robot = np.zeros(robot_shape, dtype=np.float32)
        #         hf["robot"].read_direct(robot)

        #         actions = np.zeros(actions_shape, dtype=np.float32)
        #         hf["actions"].read_direct(actions)
        #     # add to cache
        #     self._cache.put(path, (frames, robot, actions))
        # else:
        # frames, robot, actions = item
        # ep_len = frames.shape[0]
        # assert ep_len >= self._horizon, f"{ep_len}, {path}"
        # start = 0
        # end = self._horizon
        # frames_shape = list(frames.shape)
        # robot_shape = list(robot.shape)
        # actions_shape = list(actions.shape)
        frames, robot, actions, masks = self._data[index]
        ep_len = frames.shape[0]
        assert ep_len >= self._horizon, f"{ep_len}, {path}"
        start = 0
        end = self._horizon
        frames_shape = list(frames.shape)
        robot_shape = list(robot.shape)
        actions_shape = list(actions.shape)
        # if video is longer than horizon, sample a starting point
        if ep_len > self._horizon:
            offset = ep_len - self._horizon
            start = np.random.randint(0, offset + 1)
            end = start + self._horizon
            frames_shape[0] = robot_shape[0] = self._horizon
            actions_shape[0] = self._horizon - 1
        frames = frames[start:end]
        masks = masks[start:end]
        robot = robot[start:end]
        actions = actions[start : end - 1]
        # double check dimensions of data
        assert robot.shape[0] == frames.shape[0], f"{robot.shape}, {frames.shape}"
        assert robot.shape[0] - 1 == actions.shape[0], f"{robot.shape}, {actions.shape}"
        assert frames.shape[0] == self._horizon, f"{path}, {frames.shape}"
        assert frames.shape[1] == self._cf.channels, f"{path}, {frames.shape}"
        if self._cf.multiview:
            assert (
                frames.shape[2] == 2 * self._cf.image_width
            ), f"{path}, {frames.shape}"
        else:
            assert frames.shape[2] == self._cf.image_width, f"{path}, {frames.shape}"
        assert actions.shape[-1] == self._cf.action_dim, f"{path}, {actions.shape}"
        assert robot.shape[-1] == self._cf.robot_dim, f"{path}, {robot.shape}"
        return frames, robot, actions, masks

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
