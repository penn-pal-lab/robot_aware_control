import os
import random

import torch
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
    data_path = os.path.join(config.data_root, "widowx_views", "widowx1_c0")
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            high_error = any([x["high_error"] for x in motion_info[d.path]])
            if high_error:
                files.append(d.path)
                file_labels.append("widowx_widowx1_c0")

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
    file_type = "hdf5"
    files = []
    file_labels = []

    with open(config.world_error_dict, "rb") as f:
        motion_info = pickle.load(f)
    data_path = os.path.join(config.data_root, "widowx_views", "widowx1_c0")
    for d in os.scandir(data_path):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            high_error = any([x["high_error"] for x in motion_info[d.path]])
            if high_error:
                files.append(d.path)
                file_labels.append("widowx_widowx1_c0")

    files = sorted(files)
    random.seed(config.seed)
    random.shuffle(files)

    n_test = 300
    X_test = files[:n_test]
    y_test = file_labels[:n_test]
    print("loaded transfer data", len(X_test))

    augment_img = config.img_augmentation
    transfer_data = RoboNetDataset(X_test, y_test, config, augment_img=augment_img)

    transfer_loader = DataLoader(
        transfer_data,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    return transfer_loader
