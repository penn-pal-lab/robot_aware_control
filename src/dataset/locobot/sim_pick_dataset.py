import os
from src.dataset.robonet.robonet_dataset import RoboNetDataset, normalize

import h5py
import numpy as np
import torch
import torchvision.transforms as tf

class SimPickDataset(RoboNetDataset):
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
            # For robonet data, states is normalized. For locobot, states is raw.
            states = self._load_states(hf, start, end)
            actions = self._load_actions(hf, None, None, start, end - 1)
            masks = hf[MASK_KEY][start:end].astype(np.float32)
            qpos = self._load_qpos(hf, start, end)
            assert (
                len(images) == len(states) == len(actions) + 1 == len(masks)
            ), f"{hdf5_path}, {images.shape}, {states.shape}, {actions.shape}, {masks.shape}"

            # preprocessing
            images, masks = self._preprocess_images_masks(images, masks)
            # normalize and transform the states
            states = self._preprocess_states(states, None, None, robot_viewpoint, idx)
            # create the actions
            actions = self._preprocess_actions(states, actions, None, None, idx)
            if "robot" in hf.attrs:
                robot = hf.attrs["robot"]
            else:
                if "locobot" in robot_viewpoint:
                    robot = "locobot"
                elif "franka" in robot_viewpoint:
                    robot = "franka"

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
        return out

    def _load_actions(self, file_pointer, gripper_low, gripper_high, start, end):
        actions = file_pointer["actions"][:].astype(np.float32)
        return actions[start:end]

    def _load_states(self, file_pointer, start, end):
        states = file_pointer["states"][start:end].astype(np.float32)
        return states

    def _load_qpos(self, file_pointer, start, end):
        qpos = file_pointer["qpos"][start:end].astype(np.float32)
        if qpos.shape[-1] != self._config.robot_joint_dim:
            assert self._config.robot_joint_dim > qpos.shape[-1]
            pad = self._config.robot_joint_dim - qpos.shape[-1]
            qpos = np.pad(qpos, [(0, 0), (0, pad)])
        return qpos


    def _preprocess_images_masks(self, images, masks):
        """
        Converts numpy image (uint8) [0, 255] to tensor (float) [0, 1].
        """
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
        return states

    def _preprocess_actions(self, states, actions, low, high, idx):
        return torch.from_numpy(actions)

if __name__ == "__main__":
    from src.utils.plot import save_gif
    from src.config import argparser

    config, _ = argparser()
    config.data_root = "/home/pallab/locobot_ws/src/roboaware/demos/"
    config.batch_size = 16  # needs to be multiple of the # of robots
    config.video_length = 15
    config.image_width = 64
    config.image_height = 48
    config.data_threads = 0
    config.action_dim = 5
    config.robot_dim = 5
    config.robot_joint_dim = 7

    file_type = "hdf5"
    files = []
    file_labels = []

    data_path = os.path.join(config.data_root)
    for d in os.scandir(data_path):
        if d.is_file() and d.name.endswith("hdf5"):
            files.append(d.path)
            file_labels.append("locobot_c0")

    dataset = SimPickDataset(files, file_labels, config)

    for i in range(10):
        sample = dataset[i]
        import ipdb; ipdb.set_trace()


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