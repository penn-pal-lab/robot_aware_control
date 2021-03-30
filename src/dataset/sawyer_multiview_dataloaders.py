import os
import random

import ipdb
import pickle
import numpy as np
import torch
import torchvision.transforms as tf
from sklearn.model_selection import train_test_split
from src.dataset.multirobot_dataset import RobotDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets.folder import has_file_allowed_extension

# which viewpoints to train on
SAWYER_TRAIN_DIRS = ["sudri0_c0", "sudri0_c1", "sudri0_c2", "sudri2_c0", "sudri2_c2", "vestri_table2_c0", "vestri_table2_c1", "vestri_table2_c2", ]
# which viewpoints to test on
SAWYER_TEST_DIRS = ["sudri2_c1"]

def create_finetune_loaders(config):
    # finetune on unseen sawyer viewpoint
    file_type = "hdf5"
    files = []
    # motion info
    with open(config.world_error_dict, "rb") as f:
        motion_info = pickle.load(f)
    file_labels = []
    data_path = os.path.join(config.data_root, "sawyer_views", SAWYER_TEST_DIRS[0])
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            high_error = any([x["high_error"] for x in motion_info[d.path]])
            if high_error:
                files.append(d.path)
                file_labels.append("sawyer_" + SAWYER_TEST_DIRS[0])
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
    train_data = RobotDataset(X_train, y_train, config, augment_img=augment_img)
    test_data = RobotDataset(X_test, y_test, config)

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

    # create a small deterministic dataloader for comparison across runs
    # because train / test loaders have multiple workers, RNG is tricky.
    # num_gifs = min(config.batch_size, 10)
    # comp_files = X_test[:num_gifs]
    # comp_file_labels = y_test[:num_gifs]
    # # set to train so we get random snippet from videos
    # comp_data = RobotDataset(comp_files, comp_file_labels, config, load_snippet=True)
    # comp_loader = DataLoader(
    #     comp_data, num_workers=0, batch_size=num_gifs, shuffle=False
    # )
    return train_loader, test_loader


def create_transfer_loader(config):
    """
    For evaluating zero shot performance on the unseen viewpoint
    """
    file_type = "hdf5"
    files = []
    file_labels = []
    data_path = os.path.join(config.data_root, "sawyer_views", SAWYER_TEST_DIRS[0])
    # motion info
    with open(config.world_error_dict, "rb") as f:
        motion_info = pickle.load(f)
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            high_error = any([x["high_error"] for x in motion_info[d.path]])
            if high_error:
                files.append(d.path)
                file_labels.append("sawyer_" + SAWYER_TEST_DIRS[0])
    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)

    # only use 500 videos (400 training) like robonet
    files = files[:500]
    file_labels = file_labels[:500]
    print("loaded transfer data", data_path, len(files))
    split_rng = np.random.RandomState(config.seed)
    X_train, _, y_train, _ = train_test_split(
        files, file_labels, test_size=1 - config.train_val_split, random_state=split_rng
    )

    train_data = RobotDataset(X_train, y_train, config)
    loader = DataLoader(
        train_data,
        num_workers=config.data_threads,
        batch_size=config.test_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    return loader


def create_loaders(config):
    """Creates training data for Sawyer viewpoint data

    No stratification!
    Args:
        config ([type]): [description]

    Returns:
        List[DataLoader]: train, test, comparison loader of viewpoint data
    """
    # create sawyer viewpoint training data
    file_type = "hdf5"
    files = []
    file_labels = []
    data_path = os.path.join(config.data_root, "sawyer_views")
    for folder in os.scandir(data_path):
        if folder.is_dir() and folder.name in SAWYER_TRAIN_DIRS:
            for d in os.scandir(folder.path):
                if d.is_file() and has_file_allowed_extension(d.path, file_type):
                    files.append(d.path)
                    file_labels.append("sawyer_" + folder.name)

    file_and_labels = zip(files, file_labels)
    file_and_labels = sorted(file_and_labels, key=lambda x: x[0])
    random.seed(config.seed)
    random.shuffle(file_and_labels)

    files = [x[0] for x in file_and_labels]
    file_labels = [x[1] for x in file_and_labels]
    # Stratify by viewpoint
    split_rng = np.random.RandomState(config.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        files,
        file_labels,
        test_size=1 - config.train_val_split,
        random_state=split_rng,
    )
    augment_img = config.img_augmentation
    train_data = RobotDataset(X_train, y_train, config, augment_img=augment_img)
    test_data = RobotDataset(X_test, y_test, config)

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

    # create a small deterministic dataloader for comparison across runs
    # because train / test loaders have multiple workers, RNG is tricky.
    # num_gifs = min(config.batch_size, 10)
    # comp_files = X_test[:num_gifs]
    # comp_file_labels = y_test[:num_gifs]
    # comp_data = RobotDataset(comp_files, comp_file_labels, config)
    # comp_loader = DataLoader(
    #     comp_data, num_workers=0, batch_size=num_gifs, shuffle=False
    # )
    return train_loader, test_loader


# def create_loaders(config):
#     """Creates training data for Sawyer viewpoint data

#     Stratifies by viewpoint
#     Args:
#         config ([type]): [description]

#     Returns:
#         List[DataLoader]: train, test, comparison loader of viewpoint data
#     """
#     # create sawyer viewpoint training data
#     file_type = "hdf5"
#     files = []
#     file_labels = []
#     data_path = os.path.join(config.data_root, "sawyer_views")
#     for folder in os.scandir(data_path):
#         if folder.is_dir() and folder.name in SAWYER_TRAIN_DIRS:
#             for d in os.scandir(folder.path):
#                 if d.is_file() and has_file_allowed_extension(d.path, file_type):
#                     files.append(d.path)
#                     file_labels.append(folder.name)

#     file_and_labels = zip(files, file_labels)
#     file_and_labels = sorted(file_and_labels, key=lambda x: x[0])
#     random.seed(config.seed)
#     random.shuffle(file_and_labels)

#     files = [x[0] for x in file_and_labels]
#     file_labels = [x[1] for x in file_and_labels]
#     # Stratify by viewpoint
#     split_rng = np.random.RandomState(config.seed)
#     X_train, X_test, y_train, y_test = train_test_split(
#         files,
#         file_labels,
#         test_size=1 - config.train_val_split,
#         stratify=file_labels,
#         random_state=split_rng,
#     )
#     augment_img = config.img_augmentation
#     train_data = RobotDataset(X_train, y_train, config, augment_img=augment_img)
#     test_data = RobotDataset(X_test, y_test, config)
#     # stratified sampler
#     robots, counts = np.unique(file_labels, return_counts=True)
#     class_weight = {}
#     for robot, count in zip(robots, counts):
#         class_weight[robot] = count

#     print("loaded training data", class_weight)
#     # scale weights so we sample uniformly by class
#     train_weights = torch.DoubleTensor(
#         [1 / (len(robots) * class_weight[robot]) for robot in y_train]
#     )
#     train_sampler = WeightedRandomSampler(
#         train_weights,
#         len(y_train),
#         generator=torch.Generator().manual_seed(config.seed),
#     )
#     train_loader = DataLoader(
#         train_data,
#         num_workers=config.data_threads,
#         batch_size=config.batch_size,
#         shuffle=False,
#         drop_last=False,
#         pin_memory=True,
#         sampler=train_sampler,
#     )

#     test_weights = torch.DoubleTensor(
#         [1 / (len(robots) * class_weight[robot]) for robot in y_test]
#     )
#     test_sampler = WeightedRandomSampler(
#         test_weights,
#         len(y_test),
#         generator=torch.Generator().manual_seed(config.seed),
#     )
#     test_loader = DataLoader(
#         test_data,
#         num_workers=config.data_threads,
#         batch_size=config.test_batch_size,
#         shuffle=False,
#         drop_last=False,
#         pin_memory=True,
#         sampler=test_sampler,
#     )

#     # create a small deterministic dataloader for comparison across runs
#     # because train / test loaders have multiple workers, RNG is tricky.
#     num_gifs = min(config.batch_size, 10)
#     comp_files = X_test[:num_gifs]
#     comp_file_labels = y_test[:num_gifs]
#     comp_data = RobotDataset(comp_files, comp_file_labels, config)
#     comp_loader = DataLoader(
#         comp_data, num_workers=0, batch_size=num_gifs, shuffle=False
#     )
#     return train_loader, test_loader, comp_loader



def check_robot_masks(hdf5_list, target_dir):
    """
    Generates mask gifs to check accuracy
    """
    for i, traj_name in enumerate(tqdm(hdf5_list, f"visualizing masks")):
        with h5py.File(traj_name, "r") as f:
            masks = f["mask"][:]
            imgs = f["frames"][:]
        imgs[masks] = (0, 255, 255)
        gif_path = os.path.join(target_dir, f"{i}.gif")
        imageio.mimwrite(gif_path, imgs)


if __name__ == "__main__":
    import imageio
    from src.config import argparser
    from torch.multiprocessing import set_start_method
    from tqdm import tqdm
    import h5py

    config, _ = argparser()
    config.data_root = "/home/ed/Robonet"
    config.batch_size = 64  # needs to be multiple of the # of robots
    config.video_length = 31
    config.image_width = 64
    # config.impute_autograsp_action = True
    config.num_workers = 2
    config.action_dim = 5

    # check each sawyer mask folder's views and see if they are good.
    file_type = "hdf5"
    files = []
    file_labels = []
    robots = ["sawyer"]
    view_folder = "vestri_table2_c2_hdf5"
    data_path = os.path.join(config.data_root, "sawyer_views", view_folder )
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            files.append(d.path)

    files = files[:10]
    view_folder_gifs = f"{view_folder}_gifs"
    os.makedirs(view_folder_gifs, exist_ok=True)
    check_robot_masks(files, view_folder_gifs)