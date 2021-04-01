import os
import random

import ipdb
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.dataset.robonet.robonet_dataset import RoboNetDataset
from torch.utils.data import DataLoader
from torchvision.datasets.folder import has_file_allowed_extension
import pickle

def create_finetune_loaders(config):
    # finetune on baxter left data
    file_type = "hdf5"
    files = []
    file_labels = []
    # motion info
    with open(config.world_error_dict, "rb") as f:
        motion_info = pickle.load(f)
    data_path = os.path.join(config.data_root, "baxter_views", "left_c0")
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            if f"baxter_left" in d.path:
                high_error = any([x["high_error"] for x in motion_info[d.path]])
                if high_error:
                    files.append(d.path)
                    file_labels.append("baxter")

    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)

    n_test = config.finetune_num_test
    n_train = config.finetune_num_train

    X_test = files[:n_test]
    y_test = file_labels[:n_test]

    X_train = files[n_test: n_test + n_train]
    y_train = file_labels[n_test: n_test + n_train]
    print("loaded finetuning data", len(X_train) + len(X_test))

    augment_img = config.img_augmentation
    train_data = RoboNetDataset(X_train, y_train, config, augment_img=augment_img)
    test_data = RoboNetDataset(X_test, y_test, config)

    train_loader = DataLoader(
        train_data,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    test_loader = DataLoader(
        test_data,
        num_workers=config.data_threads,
        batch_size=config.test_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    return train_loader, test_loader


def create_transfer_loader(config):
    """
    For evaluating zero shot performance on the baxter data
    Contains baxter_left data with high baseline error
    """
    # finetune on baxter left data
    file_type = "hdf5"
    files = []
    file_labels = []
    data_path = os.path.join(config.data_root, "baxter_views", "left_c0")
    # motion info
    with open(config.world_error_dict, "rb") as f:
        motion_info = pickle.load(f)
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            if "baxter_left" in d.path:
                high_error = any([x["high_error"] for x in motion_info[d.path]])
                if high_error:
                    files.append(d.path)
                    file_labels.append("baxter_left")
                # files.append(d.path)
                # file_labels.append("baxter_left")

    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)

    # only use 500 videos (400 training) like robonet
    files = files[:500]
    file_labels = ["baxter"] * len(files)
    data = RoboNetDataset(files, file_labels, config)
    loader = DataLoader(
        data,
        num_workers=config.data_threads,
        batch_size=config.test_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    return loader


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

    # train, test = create_loaders(config)
    # # verify our batches have good class distribution
    # it = iter(train)

    # for i, (x, y) in enumerate(it):
    #     # robots, counts = np.unique(y, return_counts=True)
    #     # class_weight = {}
    #     # for robot, count in zip(robots, counts):
    #     #     class_weight[robot] = count / len(y)

    #     # print(class_weight)
    #     # print()
    #     imgs, states, actions, masks = x
    #     for robot_imgs, robot_masks in zip(imgs, masks):
    #         # B x C x H x W
    #         # B x H x W x C
    #         img_gif = robot_imgs.permute(0, 2, 3, 1).clamp_(0, 1).cpu().numpy()
    #         img_gif = np.uint8(img_gif * 255)
    #         robot_masks = robot_masks.cpu().numpy().squeeze().astype(bool)
    #         img_gif[robot_masks] = (0, 255, 255)
    #         imageio.mimwrite(f"test{i}_{y[0]}.gif", img_gif)
    #         break

    #     if i >= 10:
    #         break
