import numpy as np
import ipdb
import h5py
import os
import torch

from src.dataset.robonet.robonet_dataset import RoboNetDataset, create_heatmaps


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
            states = torch.from_numpy(hf["states"][start:end].astype(np.float32))
            actions = hf["actions"][start:end - 1].astype(np.float32)
            masks = hf["masks"][start:end].astype(np.float32)
            qpos = hf["qpos"][start:end].astype(np.float32)

            assert (
                len(images) == len(states) == len(actions) + 1 == len(masks)
            ), f"{hdf5_path}, {images.shape}, {states.shape}, {actions.shape}"

            # preprocessing
            images, masks = self._preprocess_images_masks(images, masks)
            folder = os.path.basename(os.path.dirname(hdf5_path))
            if self._config.model_use_heatmap:
                # since states are stored in raw coordinates, don't need to
               # denormalize.
                high = np.ones(5)
                low = np.zeros(5)
                robot = "locobot"
                heatmaps = create_heatmaps(states, low, high, robot, folder)
        out = {
            "images": images,
            "states": states,
            "actions": actions,
            "qpos": qpos,
            "masks": masks,
            "robot": "locobot",
            "folder": folder,
            "file_path": hdf5_path,
            "idx": idx,
        }
        if self._config.model_use_heatmap:
            out["heatmaps"] = heatmaps
        return out

if __name__ == "__main__":
    from src.utils.plot import save_gif
    from src.config import argparser
    import torch

    config, _ = argparser()
    config.data_root = "/media/ed/hdd/Datasets"
    config.batch_size = 16  # needs to be multiple of the # of robots
    config.video_length = 31
    config.image_width = 64
    # config.impute_autograsp_action = True
    config.data_threads = 0
    config.action_dim = 5
    config.model_use_heatmap = True

    from src.dataset.locobot.locobot_singleview_dataloader import create_loaders
    loader, _ = create_loaders(config)

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