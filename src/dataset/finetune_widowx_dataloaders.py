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
import pickle

def create_finetune_loaders(config):
    # finetune on baxter left data
    file_type = "hdf5"
    files = []
    file_labels = []
    # motion info
    # with open(config.world_error_dict, "rb") as f:
    #     motion_info = pickle.load(f)
    data_path = os.path.join(config.data_root, "widowx_views", "widowx1_c0")
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            # if f"baxter_left" in d.path:
            #     high_error = any([x["high_error"] for x in motion_info[d.path]])
            #     if high_error:
            files.append(d.path)
            file_labels.append("widowx")

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
    num_gifs = min(config.batch_size, 10)
    comp_files = [f for f in X_test[:num_gifs]]
    comp_file_labels = ["widowx"] * len(comp_files)
    # set to train so we get random snippet from videos
    comp_data = RobotDataset(comp_files, comp_file_labels, config, load_snippet=True)
    comp_loader = DataLoader(
        comp_data, num_workers=0, batch_size=num_gifs, shuffle=False
    )
    return train_loader, test_loader, comp_loader
