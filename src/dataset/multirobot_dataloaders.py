import os
import random
import ipdb
import pandas as pd

import torch
from ipdb import set_trace as st
from torch.utils.data import random_split
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.dataloader import DataLoader

from src.dataset.multirobot_dataset import (
    BaxterDataset,
    MultiRobotDataset,
    SawyerDataset,
    WidowXDataset,
)


def create_split(config):
    metadata_path = os.path.join(config.data_root, "meta_data.pkl")
    metadata = pd.read_pickle(metadata_path, compression="gzip")

    # baxter files
    baxter_dataset = BaxterDataset(metadata, config)
    if config.preload_ram:
        baxter_dataset.preload_ram()
    train_num = int(len(baxter_dataset) * config.train_val_split)
    val_num = len(baxter_dataset) - train_num
    lengths = [train_num, val_num]
    baxter_train, baxter_val = random_split(
        baxter_dataset, lengths, generator=torch.Generator().manual_seed(config.seed)
    )


    # widowx files
    widowx_dataset = WidowXDataset(metadata, config)
    if config.preload_ram:
        widowx_dataset.preload_ram()
    train_num = int(len(widowx_dataset) * config.train_val_split)
    val_num = len(widowx_dataset) - train_num
    lengths = [train_num, val_num]
    widowx_train, widowx_val = random_split(
        widowx_dataset, lengths, generator=torch.Generator().manual_seed(config.seed)
    )
    if config.preload_ram:
        widowx_train.preload_ram()
        widowx_val.preload_ram()

    # sawyer files
    sawyer_dataset = SawyerDataset(metadata, config)
    if config.preload_ram:
        sawyer_dataset.preload_ram()
    train_num = int(len(sawyer_dataset) * config.train_val_split)
    val_num = len(sawyer_dataset) - train_num
    lengths = [train_num, val_num]
    sawyer_train, sawyer_val = random_split(
        sawyer_dataset, lengths, generator=torch.Generator().manual_seed(config.seed)
    )
    if config.preload_ram:
        sawyer_train.preload_ram()
        sawyer_val.preload_ram()


    train_robots = [baxter_train, widowx_train, sawyer_train]
    train_data = MultiRobotDataset(config, train_robots)
    val_data = ConcatDataset([baxter_val, widowx_val, sawyer_val])
    return train_data, val_data


def create_loaders(config):
    train_data, test_data = create_split(config)
    train_loader = train_data # multirobot already has iter implemented
    test_loader = DataLoader(test_data,
                num_workers=config.data_threads,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True)
    return train_loader, test_loader


def get_batch(loader, device):
    while True:
        for data in loader:
            # transpose from (B, L, C, W, H) to (L, B, C, W, H)
            imgs, states, actions, masks = data
            frames = imgs.transpose_(1, 0).to(device)
            robots = states.transpose_(1, 0).to(device)
            actions = actions.transpose_(1, 0).to(device)
            masks = masks.transpose_(1, 0).to(device)
            yield frames, robots, actions, masks


if __name__ == "__main__":
    import torch
    from src.config import argparser

    config, _ = argparser()
    config.data_root = "/home/ed/Robonet/hdf5"
    config.batch_size = 3 * 6  # needs to be multiple of the # of robots
    config.video_length = 31
    config.image_width = 64
    config.impute_autograsp_action = True
    config.action_dim = 5
    loader = MultiRobotDataset(config)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for i, data in enumerate(get_batch(loader, device)):
        print("foo")
        if i >= 2:
            break
