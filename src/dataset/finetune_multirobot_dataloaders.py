import os
import random

import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from src.dataset.multirobot_dataset import RobotDataset
from torchvision.datasets.folder import has_file_allowed_extension
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import random_split
import torch
import ipdb


def create_finetune_loaders(config):
    # finetune on baxter left data
    file_type = "hdf5"
    files = []
    file_labels = []
    robots = ["baxter"]
    for d in os.scandir(config.data_root):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            for r in robots:
                if f"{r}_left" in d.path:
                    files.append(d.path)
                    file_labels.append(r)
                    break
    # only use 500 videos (400 training) like robonet
    random.seed(config.seed)
    random.shuffle(files)
    files = files[:500]
    file_labels = ["baxter"] * 500
    data = RobotDataset(files, file_labels, config)
    train_num = int(len(data) * config.train_val_split)
    val_num = len(data) - train_num
    lengths = [train_num, val_num]
    train, test = random_split(
        data, lengths=lengths, generator=torch.Generator().manual_seed(config.seed)
    )

    train_loader = DataLoader(
        train,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )
    return train_loader, test_loader


def create_loaders(config):
    # create sawyer training data
    file_type = "hdf5"
    files = []
    file_labels = []
    robots = ["sawyer"]
    for d in os.scandir(config.data_root):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            for r in robots:
                if r in d.path:
                    files.append(d.path)
                    file_labels.append(r)
                    break

    X_train, X_test, y_train, y_test = train_test_split(
        files, file_labels, test_size=1 - config.train_val_split, stratify=file_labels
    )
    train_data = RobotDataset(X_train, y_train, config)
    test_data = RobotDataset(X_test, y_test, config)
    # stratified sampler
    robots, counts = np.unique(file_labels, return_counts=True)
    class_weight = {}
    for robot, count in zip(robots, counts):
        class_weight[robot] = count
    # scale weights so we sample uniformly by class
    train_weights = torch.DoubleTensor(
        [1 / (len(robots) * class_weight[robot]) for robot in y_train]
    )
    train_sampler = WeightedRandomSampler(
        train_weights,
        len(y_train),
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(
        train_data,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_weights = torch.DoubleTensor(
        [1 / (len(robots) * class_weight[robot]) for robot in y_test]
    )
    test_sampler = WeightedRandomSampler(
        test_weights,
        len(y_test),
        generator=torch.Generator().manual_seed(config.seed),
    )
    test_loader = DataLoader(
        test_data,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        sampler=test_sampler,
    )
    return train_loader, test_loader


def get_batch(loader, device):
    while True:
        for data, _ in loader:
            # transpose from (B, L, C, W, H) to (L, B, C, W, H)
            imgs, states, actions, masks = data
            frames = imgs.transpose_(1, 0).to(device)
            robots = states.transpose_(1, 0).to(device)
            actions = actions.transpose_(1, 0).to(device)
            masks = masks.transpose_(1, 0).to(device)
            yield frames, robots, actions, masks


if __name__ == "__main__":
    from src.config import argparser
    from torch.multiprocessing import set_start_method
    import imageio

    set_start_method("spawn")
    config, _ = argparser()
    config.data_root = "/home/ed/new_hdf5"
    config.batch_size = 64  # needs to be multiple of the # of robots
    config.video_length = 31
    config.image_width = 64
    # config.impute_autograsp_action = True
    config.num_workers = 2
    config.action_dim = 5

    train, test = create_loaders(config)
    # verify our batches have good class distribution
    it = iter(train)

    for i, (x, y) in enumerate(it):
        # robots, counts = np.unique(y, return_counts=True)
        # class_weight = {}
        # for robot, count in zip(robots, counts):
        #     class_weight[robot] = count / len(y)

        # print(class_weight)
        # print()
        imgs, states, actions, masks = x
        for robot_imgs, robot_masks in zip(imgs, masks):
            # B x C x H x W
            # B x H x W x C
            img_gif = robot_imgs.permute(0, 2, 3, 1).clamp_(0, 1).cpu().numpy()
            img_gif = np.uint8(img_gif * 255)
            robot_masks = robot_masks.cpu().numpy().squeeze().astype(bool)
            img_gif[robot_masks] = (0, 255, 255)
            imageio.mimwrite(f"test{i}_{y[0]}.gif", img_gif)
            break

        if i >= 10:
            break
