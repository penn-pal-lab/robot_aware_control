import numpy as np
import ipdb
import h5py
import os

from src.dataset.robonet.robonet_dataset import RoboNetDataset


class LocobotDataset(RoboNetDataset):
    def __init__(
        self, hdf5_list, robot_list, config, augment_img=False, load_snippet=False
    ):
        super().__init__(hdf5_list, robot_list, config, augment_img=augment_img, load_snippet=load_snippet)

    def __getitem__(self, idx):
        """
        Opens the hdf5 file and extracts the videos, actions, states, and masks
        """
        if idx in self._memory:
            return self._memory[idx]

        hdf5_path = self._traj_names[idx]
        with h5py.File(hdf5_path, "r") as hf:
            ep_len = hf["observations"].shape[0]
            assert ep_len >= self._video_length, f"{ep_len}, {hdf5_path}"
            start = 0
            end = ep_len
            if ep_len > self._video_length:
                offset = ep_len - self._video_length
                start = self._rng.randint(0, offset + 1)
                end = start + self._video_length

            images = hf["observations"][start:end]
            states = hf["states"][start:end].astype(np.float32)
            actions = hf["actions"][start:end - 1].astype(np.float32)
            masks = hf["masks"][start:end].astype(np.float32)
            qpos = hf["qpos"][start:end].astype(np.float32)

            assert (
                len(images) == len(states) == len(actions) + 1 == len(masks)
            ), f"{hdf5_path}, {images.shape}, {states.shape}, {actions.shape}"

            # preprocessing
            images, masks = self._preprocess_images_masks(images, masks)
            folder = os.path.basename(os.path.dirname(hdf5_path))
        out = {
            "images": images,
            "states": states,
            "actions": actions,
            "qpos": qpos,
            "masks": masks,
            "robot": "LoCoBot",
            "folder": folder,
            "file_path": hdf5_path,
            "idx": idx,
        }
        return out
