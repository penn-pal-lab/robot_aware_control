""" Find videos with object interaction in the dataset """
import os
import pickle
from collections import defaultdict
from math import floor

import imageio
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from src.config import argparser
from src.dataset.robonet.robonet_dataset import RoboNetDataset
from src.prediction.losses import world_mse_criterion
from src.prediction.models.dynamics import CopyModel
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.folder import has_file_allowed_extension
from tqdm import tqdm
import torch


def eval_step(data, model: CopyModel, cf):
    """Evaluates the error metric on the data"""
    x, masks, _ = data
    losses = defaultdict(float)
    for i in range(1, cf.n_past + cf.n_future):
        x_j = x[i - 1]
        x_i, m_i = x[i], masks[i]
        x_pred = model(x_j, None, x_i, m_i)
        # compute world loss error
        world_mse = world_mse_criterion(x_pred, x_i, m_i)
        losses["world_mse"] += world_mse
    return losses

def plot_error(loader, model, name):
    """ Visualize histogram of the copy error over the data"""
    all_err = []
    for d in tqdm(loader, desc="plot error"):
        x = d["images"].transpose_(0, 1)  # T x B
        masks = d["masks"].transpose_(0, 1)  # T x B
        names = d["folder"]  # B
        batch = x, masks, names
        losses = eval_step(batch, model, config)
        all_err.append(losses["world_mse"])

    # histogram of world err
    all_err = np.asarray(all_err)
    print("Mean world error", all_err.mean())
    plt.hist(all_err, bins=24, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(all_err.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.title(name)
    max_ylim = plt.ylim()[1]
    plt.text(all_err.mean()*1.1, max_ylim*0.9, 'Mean: {:.3f}'.format(all_err.mean()))
    plt.savefig(f"{name}_histogram.png")
    plt.show()

def make_above_threshold_videos(loader, model, threshold, max_gifs):
    """ Create GIFs of the videos that are above the defined error threshold."""
    all_err = []
    gifs_created = 0
    for d in tqdm(loader, desc="make gifs"):
        x = d["images"].transpose_(0, 1)  # T x B
        masks = d["masks"].transpose_(0, 1)  # T x B
        names = d["folder"]  # B
        batch = x, masks, names
        losses = eval_step(batch, model, config)
        all_err.append(losses["world_mse"])
        if losses["world_mse"].item() > threshold and gifs_created < max_gifs:
            mask = batch[1]
            robot_mask = mask.type(torch.bool)
            robot_mask = robot_mask.repeat(1,1,3,1,1)
            batch[0][robot_mask] *= 0
            imgs = (255 * batch[0].squeeze().permute(0,2,3,1)).cpu().numpy().astype(np.uint8)
            gif_name = f"{names[0]}_{losses['world_mse']:.6f}.gif"
            imageio.mimwrite(gif_name, imgs)
            gifs_created += 1
        if gifs_created >= max_gifs:
            break

def make_metadata(loader, model, threshold, write_path):
    """ make metadata dictionary to contain error metric info for training"""
    pass
    """
    Meta dictionary that stores information about each sequence.
    Key: video name
    Value: List of dictionaries containing
            - start / end of sequence
            - cost of sequence
    """
    meta_dict = defaultdict(bool)
    num_high_err_videos = 0
    for d in tqdm(loader, desc="make metadata"):
        x = d["images"].transpose_(0, 1)  # T x B
        masks = d["masks"].transpose_(0, 1)  # T x B
        names = d["folder"]  # B
        paths = d["file_path"]
        high_err_video = False
        batch = x, masks, names
        losses = eval_step(batch, model, config)
        if losses["world_mse"].item() >= threshold:
            high_err_video = True
            num_high_err_videos += 1
        meta_dict[paths[0]] = high_err_video

    print(f"above threshold videos: {num_high_err_videos}")
    # save meta dict
    with open(write_path, "wb") as f:
        pickle.dump(meta_dict, f)

def measure_obj_movement(ROBOT, VIEWPOINT_FOLDER, config):
    """
    Generate meta dictionary for recording baseline error per video
    1. Load all videos
    2. Use copy model to get error metrics of the videos
    3. Visualize the error in a histogram to pick object interaction threshold
    4. Save some GIFs of the object interaction threshold to verify
    5. Generate metadata file for the robot viewpoint. Print some statistics about the metadata like how many videos are above the threshold.
    """
    window = config.n_past + config.n_future

    MAX_VIDEOS = 100000000
    DATA_PATH = os.path.join(config.data_root, f"{ROBOT}_views", VIEWPOINT_FOLDER)
    MAX_GIFS = 10
    METADATA_PATH = os.path.join(DATA_PATH, "obj_movement.pkl")

    files = []
    file_labels = []
    count = 0
    for d in os.scandir(DATA_PATH):
        if d.is_file() and has_file_allowed_extension(d.path, "hdf5"):
            files.append(d.path)
            file_labels.append(f"{ROBOT}_{VIEWPOINT_FOLDER}")
            count += 1
            if count >= MAX_VIDEOS:
                break

    dataset = RoboNetDataset(files, file_labels, config)
    loader = DataLoader(
        dataset,
        num_workers=config.data_threads,
        batch_size=config.batch_size,
        pin_memory=True,
    )
    model = CopyModel()
    # plot_error(loader, model, ROBOT + "_" + VIEWPOINT_FOLDER)
    """ Set some thresholds to inf since their viewpoints are bad"""
    THRESHOLDS = {
        "sawyer_sudri0_c0": 0.114,
        "sawyer_sudri0_c1": 0.21, # overhead view
        "sawyer_sudri0_c2": 0.18,
        "sawyer_vestri_table2_c0": 0.09,
        "sawyer_vestri_table2_c1": 0.26, # overhead view, looks bad
        "sawyer_vestri_table2_c2": 0.149,
        "sawyer_sudri2_c0": 0.095,
        "sawyer_sudri2_c1": 0.223,
        "sawyer_sudri2_c2": 0.165,
        "baxter_left_c0": 0.017,
        "widowx_widowx1_c0": 0.11,
        "locobot_c0": 0.15,
    }
    threshold = THRESHOLDS[ROBOT + "_" + VIEWPOINT_FOLDER]
    # make_above_threshold_videos(loader, model, threshold, MAX_GIFS)
    make_metadata(loader, model, threshold, METADATA_PATH)
    # print("total videos", len(dataset))

if __name__ == "__main__":
    config, _ = argparser()
    config.data_root = "/scratch/anonymous/Robonet"
    config.batch_size = 1
    config.video_length = 31
    config.image_width = 64
    # config.impute_autograsp_action = True
    config.num_workers = 4
    config.action_dim = 5
    config.n_past = 1
    config.n_future = 30

    all_robots = ["sawyer", "baxter", "widowx"]
    robot_viewpoints = {
        "baxter" : ["left_c0"],
        "widowx" : ["widowx1_c0"],
        "sawyer" : ["sudri0_c0", "sudri0_c1", "sudri0_c2", "sudri2_c0", "sudri2_c1", "sudri2_c2", "vestri_table2_c0", "vestri_table2_c1", "vestri_table2_c2"],
        "locobot": ["c0"],
    }

    for robot, viewpoints in robot_viewpoints.items():
        for vp in viewpoints:
            measure_obj_movement(robot, vp, config)