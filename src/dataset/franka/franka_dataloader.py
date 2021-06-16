import os
import random
import ipdb

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import has_file_allowed_extension

from src.dataset.robonet.robonet_dataset import RoboNetDataset


def create_transfer_loader(config):
    file_type = "hdf5"
    files = []
    file_labels = []

    data_path = os.path.join(config.data_root, "franka_views", "c0")
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            files.append(d.path)
            file_labels.append("franka_views")

    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)

    n_train = 400
    X_train = files[:n_train]
    y_train = file_labels[: n_train]
    print("loaded franka transfer data", len(X_train))

    augment_img = config.img_augmentation
    train_data = RoboNetDataset(X_train, y_train, config, augment_img=augment_img)

    train_loader = DataLoader(
        train_data,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    return train_loader


if __name__ == "__main__":
    from src.config import argparser

    config, _ = argparser()
    config.data_root = "/home/pallab/locobot_ws/src/eef_control/data"
    config.batch_size = 1
    config.video_length = 31
    config.image_width = 64
    config.data_threads = 0
    config.action_dim = 5
    loader = create_transfer_loader(config)
    for data in loader:
        images = data["images"]
        states = data["states"]
        print(data["actions"])
