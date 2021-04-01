import os
import random

import ipdb
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.dataset.robonet.robonet_dataset import RoboNetDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets.folder import has_file_allowed_extension

BAXTER_TRAIN_DIRS = ["left_c0"]
BAXTER_TEST_DIRS = []
WIDOWX_TRAIN_DIRS = ["widowx1_c0"]
WIDOWX_TEST_DIRS = []
SAWYER_TRAIN_DIRS = ["sudri0_c0", "sudri0_c1", "sudri0_c2", "sudri2_c0", "sudri2_c1", "sudri2_c2", "vestri_table2_c0", "vestri_table2_c1", "vestri_table2_c2"]
SAWYER_TEST_DIRS = []


def create_loaders(config):
    """
    Creates non-stratified dataset of robots.
    We assume the dataset is a folder of robot folders. Each robot folder has a viewpoint folder containing trajectories of that viewpoint.
    sawyer/
        view0/
            traj0.hdf5
            traj1.hdf5
        view1/
    widowx/
        view0/
        view1/
    baxter/
        view0/
    """
    baxter_fl = get_baxter_data(config)
    widowx_fl = get_widowx_data(config)
    sawyer_fl = get_sawyer_data(config)

    print("baxter data", len(baxter_fl))
    print("widowx data", len(widowx_fl))
    print("sawyer data", len(sawyer_fl))

    file_and_labels = baxter_fl + widowx_fl + sawyer_fl
    random.seed(config.seed)
    random.shuffle(file_and_labels)

    files = [x[0] for x in file_and_labels]
    file_labels = [x[1] for x in file_and_labels]

    split_rng = np.random.RandomState(config.seed)
    # >>>>>>>>>>> Normal sampling
    X_train, X_test, y_train, y_test = train_test_split(
        files,
        file_labels,
        test_size=1 - config.train_val_split,
        random_state=split_rng,
    )
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

    # >>>>>>>>>>>>>> Stratified Sampling
    # X_train, X_test, y_train, y_test = train_test_split(
    #     files,
    #     file_labels,
    #     test_size=1 - config.train_val_split,
    #     stratify=file_labels,
    #     random_state=split_rng,
    # )
    # augment_img = config.img_augmentation
    # train_data = RoboNetDataset(X_train, y_train, config, augment_img=augment_img)
    # test_data = RoboNetDataset(X_test, y_test, config)
    # # stratified sampler
    # robots, counts = np.unique(file_labels, return_counts=True)
    # class_weight = {}
    # for robot, count in zip(robots, counts):
    #     class_weight[robot] = count
    # # scale weights so we sample uniformly by class
    # train_weights = torch.DoubleTensor(
    #     [1 / (len(robots) * class_weight[robot]) for robot in y_train]
    # )
    # train_sampler = WeightedRandomSampler(
    #     train_weights,
    #     len(y_train),
    #     generator=torch.Generator().manual_seed(config.seed),
    # )
    # train_loader = DataLoader(
    #     train_data,
    #     num_workers=config.data_threads,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     drop_last=True,
    #     pin_memory=True,
    #     sampler=train_sampler,
    # )

    # test_weights = torch.DoubleTensor(
    #     [1 / (len(robots) * class_weight[robot]) for robot in y_test]
    # )
    # test_sampler = WeightedRandomSampler(
    #     test_weights,
    #     len(y_test),
    #     generator=torch.Generator().manual_seed(config.seed),
    # )
    # test_loader = DataLoader(
    #     test_data,
    #     num_workers=config.data_threads,
    #     batch_size=config.test_batch_size,
    #     shuffle=False,
    #     drop_last=True,
    #     pin_memory=True,
    #     sampler=test_sampler,
    # )
    # return train_loader, test_loader


def get_baxter_data(config):
    """
    Load all the robonet baxter data.
    For the robonet baxter data, there are two types of videos, baxter_left and baxter_right
    that correspond to which arm was recorded.
    Each arm has 3 cameras.

    Returns a list of hdf5 files, and a list of the viewpoint names
    """
    file_type = "hdf5"
    files = []
    file_labels = []
    data_path = os.path.join(config.data_root, "baxter_views")
    for folder in os.scandir(data_path):
        if folder.is_dir() and folder.name in BAXTER_TRAIN_DIRS:
            for d in os.scandir(folder.path):
                if d.is_file() and has_file_allowed_extension(d.path, file_type):
                    files.append(d.path)
                    file_labels.append(f"baxter_{folder.name}")

    file_and_labels = zip(files, file_labels)
    file_and_labels = sorted(file_and_labels, key=lambda x: x[0])
    random.seed(config.seed)
    random.shuffle(file_and_labels)
    return file_and_labels

def get_widowx_data(config):
    """
    Load all the robonet widowx data.
    For the robonet widowx data, there are 5 types of widowx videos.
    Returns a list of hdf5 files, and a list of the viewpoint names
    """
    file_type = "hdf5"
    files = []
    file_labels = []
    data_path = os.path.join(config.data_root, "widowx_views")
    for folder in os.scandir(data_path):
        if folder.is_dir() and folder.name in WIDOWX_TRAIN_DIRS:
            for d in os.scandir(folder.path):
                if d.is_file() and has_file_allowed_extension(d.path, file_type):
                    files.append(d.path)
                    file_labels.append(f"widowx_{folder.name}")

    file_and_labels = zip(files, file_labels)
    file_and_labels = sorted(file_and_labels, key=lambda x: x[0])
    random.seed(config.seed)
    random.shuffle(file_and_labels)
    return file_and_labels

def get_sawyer_data(config):
    """
    Load all the robonet sawyer data.
    For the robonet sawyer data, there are 5 types of sawyer videos.
    Returns a list of hdf5 files, and a list of the viewpoint names
    """
    file_type = "hdf5"
    files = []
    file_labels = []
    data_path = os.path.join(config.data_root, "sawyer_views")
    for folder in os.scandir(data_path):
        if folder.is_dir() and folder.name in SAWYER_TRAIN_DIRS:
            for d in os.scandir(folder.path):
                if d.is_file() and has_file_allowed_extension(d.path, file_type):
                    files.append(d.path)
                    file_labels.append(f"sawyer_{folder.name}")

    file_and_labels = zip(files, file_labels)
    file_and_labels = sorted(file_and_labels, key=lambda x: x[0])
    random.seed(config.seed)
    random.shuffle(file_and_labels)
    return file_and_labels



if __name__ == "__main__":
    import imageio
    from src.config import argparser
    from torch.multiprocessing import set_start_method

    set_start_method("spawn")
    config, _ = argparser()
    config.data_root = "/scratch/edward/Robonet"
    config.batch_size = 64  # needs to be multiple of the # of robots
    config.video_length = 31
    config.image_width = 64
    # config.impute_autograsp_action = True
    config.num_workers = 0
    config.action_dim = 5

    train, test = create_loaders(config)
    # verify our batches have good class distribution
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
