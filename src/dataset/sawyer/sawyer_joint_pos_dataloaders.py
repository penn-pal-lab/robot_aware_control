import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.dataset.joint_pos_dataset import JointPosDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets.folder import has_file_allowed_extension

def create_loaders(config):
    """Create training data for Sawyer->Baxter finetuning
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
    data_path = os.path.join(config.data_root, "sawyer_views")
    for folder in os.scandir(data_path):
        if folder.is_dir():
            for d in os.scandir(folder.path):
                if d.is_file() and has_file_allowed_extension(d.path, file_type):
                        files.append(d.path)
                        file_labels.append(folder.name)

    file_and_labels = zip(files, file_labels)
    file_and_labels = sorted(file_and_labels, key=lambda x: x[0])
    random.seed(config.seed)
    random.shuffle(file_and_labels)

    files = [x[0] for x in file_and_labels]
    file_labels = [x[1] for x in file_and_labels]

    # stratify train / test split by viewpoint
    split_rng = np.random.RandomState(config.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        files,
        file_labels,
        test_size=1 - config.train_val_split,
        stratify=file_labels,
        random_state=split_rng,
    )
    augment_img = config.img_augmentation
    train_data = JointPosDataset(X_train, y_train, config, augment_img=augment_img)
    test_data = JointPosDataset(X_test, y_test, config)
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
        drop_last=False,
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
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        sampler=test_sampler,
    )
    return train_loader, test_loader



def create_joint_pos_loaders(config):
    """Create training data for learning sawyer joint pos
    Dataset consists of different sawyer viewpoints
    Args:
        config (Namespace): config dict
    Returns:
        List[Dataloader]: train, test, comparison loaders
    """
    # create sawyer training data
    file_type = "hdf5"
    files = []
    file_labels = []
    data_path = os.path.join(config.data_root, "sawyer_views")
    for folder in os.scandir(data_path):
        if folder.is_dir():
            for d in os.scandir(folder.path):
                if d.is_file() and has_file_allowed_extension(d.path, file_type):
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
    print("loaded joint pos data", len(X_train) + len(X_test))
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