import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.dataset.joint_pos_dataset import JointPosDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets.folder import has_file_allowed_extension
import pickle

def create_loaders(config):
    """Create training data for Sawyer->WidowX finetuning
    Dataset consists of different sawyer viewpoints
    Stratified train / test split by sawyer viewpoint
    Imbalanced viewpoint sampling

    Args:
        config (Namespace): config dict
    Returns:
        List[Dataloader]: train, test, comparison loaders
    """
    # create sawyer training data
    file_type = "hdf5"
    files = []
    file_labels = []
    # motion info
    # with open(config.world_error_dict, "rb") as f:
    #     motion_info = pickle.load(f)
    data_path = os.path.join(config.data_root, "widowx_views")
    for folder in os.scandir(data_path):
        if folder.is_dir():
            for d in os.scandir(folder.path):
                if d.is_file() and has_file_allowed_extension(d.path, file_type):
                    # high_error = any([x["high_error"] for x in motion_info[d.path]])
                    # if high_error:
                    files.append(d.path)
                    file_labels.append(folder.name)

    file_and_labels = zip(files, file_labels)
    file_and_labels = sorted(file_and_labels, key=lambda x: x[0])
    random.seed(config.seed)
    random.shuffle(file_and_labels)

    files = [x[0] for x in file_and_labels]
    file_labels = [x[1] for x in file_and_labels]

    n_test = config.finetune_num_test
    n_train = config.finetune_num_train

    X_test = files[:n_test]
    y_test = file_labels[:n_test]

    X_train = files[n_test: n_test + n_train]
    y_train = file_labels[n_test: n_test + n_train]
    print("loaded finetuning data", len(X_train) + len(X_test))

    augment_img = config.img_augmentation
    train_data = JointPosDataset(X_train, y_train, config, augment_img=augment_img)
    test_data = JointPosDataset(X_test, y_test, config)

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