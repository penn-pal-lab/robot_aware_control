import os
import random

import ipdb
import numpy as np
import torch
import torchvision.transforms as tf
from sklearn.model_selection import train_test_split
from src.dataset.multirobot_dataset import RobotDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets.folder import has_file_allowed_extension


def create_finetune_loaders(config):
    # finetune on baxter left data
    file_type = "hdf5"
    files = []
    file_labels = []
    robots = ["baxter"]
    data_path = os.path.join(config.data_root, "new_hdf5")
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            for r in robots:
                if f"{r}_left" in d.path:
                    files.append(d.path)
                    file_labels.append(r)
                    break
    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)

    # only use 500 videos (400 training) like robonet
    files = files[:500]
    file_labels = ["baxter"] * len(files)
    split_rng = np.random.RandomState(config.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        files, file_labels, test_size=1 - config.train_val_split, random_state=split_rng
    )
    augment_img = config.img_augmentation
    train_data = RobotDataset(
        X_train, y_train, config, augment_img=augment_img, load_snippet=True
    )
    test_data = RobotDataset(X_test, y_test, config)

    train_loader = DataLoader(
        train_data,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        num_workers=config.data_threads,
        batch_size=config.test_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    # create a small deterministic dataloader for comparison across runs
    # because train / test loaders have multiple workers, RNG is tricky.
    num_gifs = min(config.batch_size, 10)
    comp_files = [f for f in X_test[:num_gifs]]
    comp_file_labels = ["baxter"] * len(comp_files)
    # set to train so we get random snippet from videos
    comp_data = RobotDataset(comp_files, comp_file_labels, config, load_snippet=True)
    comp_loader = DataLoader(
        comp_data, num_workers=0, batch_size=num_gifs, shuffle=False
    )
    return train_loader, test_loader, comp_loader


def create_transfer_loader(config):
    """
    For evaluating zero shot performance on the transfer set
    """
    # finetune on baxter left data
    file_type = "hdf5"
    files = []
    file_labels = []
    robots = ["baxter"]
    data_path = os.path.join(config.data_root, "new_hdf5")
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            for r in robots:
                if f"{r}_left" in d.path:
                    files.append(d.path)
                    file_labels.append(r)
                    break
    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)

    # only use 500 videos (400 training) like robonet
    files = files[:500]
    file_labels = ["baxter"] * len(files)
    data = RobotDataset(files, file_labels, config)
    loader = DataLoader(
        data,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    return loader


def create_loaders(config):
    # create sawyer training data
    file_type = "hdf5"
    files = []
    file_labels = []
    robots = ["sawyer"]
    data_path = os.path.join(config.data_root, "sawyer_views")
    for folder in os.scandir(data_path):
        if folder.is_dir():
            print(folder.name)
            for d in os.scandir(folder.path):
                if d.is_file() and has_file_allowed_extension(d.path, file_type):
                    for r in robots:
                        if r in d.path:
                            files.append(d.path)
                            file_labels.append(r)
                            break

    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)
    # for now, with 1 robot, stratify doesn't do anything.
    split_rng = np.random.RandomState(config.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        files,
        file_labels,
        test_size=1 - config.train_val_split,
        stratify=file_labels,
        random_state=split_rng,
    )
    augment_img = config.img_augmentation
    train_data = RobotDataset(X_train, y_train, config, augment_img=augment_img, load_snippet=True)
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

    # create a small deterministic dataloader for comparison across runs
    # because train / test loaders have multiple workers, RNG is tricky.
    num_gifs = min(config.batch_size, 10)
    comp_files = X_test[:num_gifs]
    comp_file_labels = ["baxter"] * len(comp_files)
    comp_data = RobotDataset(comp_files, comp_file_labels, config)
    comp_loader = DataLoader(
        comp_data, num_workers=0, batch_size=num_gifs, shuffle=False
    )
    return train_loader, test_loader, comp_loader

def get_batch(loader, device):
    while True:
        for data, robot_name in loader:
            # transpose from (B, L, C, W, H) to (L, B, C, W, H)
            imgs, states, actions, masks = data
            frames = imgs.transpose_(1, 0).to(device)
            robots = states.transpose_(1, 0).to(device)
            actions = actions.transpose_(1, 0).to(device)
            masks = masks.transpose_(1, 0).to(device)
            yield (frames, robots, actions, masks), robot_name


if __name__ == "__main__":
    import imageio
    from src.config import argparser
    from torch.multiprocessing import set_start_method

    set_start_method("spawn")
    config, _ = argparser()
    config.data_root = "/home/ed/"
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
