""" Find videos with object interaction in the dataset """
import os
import pickle
import random
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

def visualize_error(loader, model, name):
    """ Visualize histogram of the copy error over the data"""
    all_err = []
    for d in loader:
        x = d["images"].transpose_(0, 1)  # T x B
        masks = d["masks"].transpose_(0, 1)  # T x B
        names = d["folder"]  # B
        T = x.shape[0]
        for i in range(floor(T / window)):
            s = i * window
            e = (i + 1) * window
            batch = x[s:e], masks[s:e], names
            losses = eval_step(batch, model, config)
            all_err.append(losses["world_mse"])
            pbar.update()

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

if __name__ == "__main__":
    """
    Generate meta dictionary for recording baseline error per video
    1. Load all videos
    2. Use copy model to get error metrics of the videos
    3. Visualize the error in a histogram to pick object interaction threshold
    4. Save some GIFs of the object interaction threshold to verify
    5. Generate metadata file for the robot viewpoint.
    """
    config, _ = argparser()
    config.data_root = "/home/ed/Robonet"
    config.batch_size = 1
    config.video_length = 31
    config.image_width = 64
    # config.impute_autograsp_action = True
    config.num_workers = 4
    config.action_dim = 5
    config.n_past = 1
    config.n_future = 5

    MAX_VIDEOS = 1000
    ROBOT = "sawyer"
    VIEWPOINT_FOLDER = "sudri0_c0"
    files = []
    file_labels = []
    data_path = os.path.join(config.data_root, f"{ROBOT}_views", VIEWPOINT_FOLDER)
    count = 0
    for d in os.scandir(data_path):
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
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    model = CopyModel()
    window = config.n_past + config.n_future
    T = config.video_length
    pbar = tqdm(
        initial=0, total=len(dataset) * (floor(T / window)), desc="processing videos"
    )

    visualize_error(loader, model, ROBOT + "_" + VIEWPOINT_FOLDER)

    # num_high_err_videos = 0
    # num_high_err_snippets = 0
    # num_snippets = 0

    # num_gifs = 0
    # max_gifs = 20

    # """
    # Meta dictionary that stores information about each sequence.
    # Key: video name
    # Value: List of dictionaries containing
    #         - start / end of sequence
    #         - cost of sequence
    # """
    # meta_dict = defaultdict(list)
    # err_threshold = 0.06
    # all_err = []
    # for d in loader:
    #     x = d["images"].transpose_(0, 1)  # T x B
    #     masks = d["masks"].transpose_(0, 1)  # T x B
    #     names = d["folder"]  # B
    #     T = x.shape[0]
    #     all_losses = defaultdict(float)
    #     high_err_video = False
    #     for i in range(floor(T / window)):
    #         video_name = names[0]
    #         s = i * window
    #         e = (i + 1) * window
    #         batch = x[s:e], masks[s:e], names
    #         losses = eval_step(batch, model, config)
    #         num_snippets += 1
    #         if losses["world_mse"].item() > err_threshold:
    #             high_err_video = True
    #             num_high_err_snippets += 1
    #         entry = {
    #             "start": s,
    #             "end": e,
    #             "world_mse": losses["world_mse"].item(),
    #             "high_error": losses["world_mse"].item() > err_threshold,
    #         }
    #         meta_dict[video_name].append(entry)
    #         # if losses["world_mse"] > err_threshold and num_gifs < max_gifs:
    #         #     # save video
    #         #     mask = batch[1]
    #         #     robot_mask = mask.type(torch.bool)
    #         #     robot_mask = robot_mask.repeat(1,1,3,1,1)
    #         #     batch[0][robot_mask] *= 0
    #         #     imgs = (255 * batch[0].squeeze().permute(0,2,3,1)).cpu().numpy().astype(np.uint8)
    #         #     name = names[0].split("/")[-1][:-5]
    #         #     err = losses["world_mse"]
    #         #     gif_name = f"{name}_{losses['world_mse']:.6f}.gif"
    #         #     imageio.mimwrite(gif_name, imgs)
    #         #     num_gifs += 1
    #         # if num_gifs >= max_gifs:
    #         #     break
    #         all_err.append(losses["world_mse"])
    #         pbar.update()
    #     if num_gifs >= max_gifs:
    #         break
    #     if high_err_video:
    #         num_high_err_videos += 1
    # print(f"number of videos with high errors {num_high_err_videos}/{len(dataset)}")
    # print(
    #     f"number of snippets with high errror: {num_high_err_snippets}/{num_snippets}"
    # )

    # # histogram of world err
    # # all_err = np.asarray(all_err)
    # # print(all_err.mean())
    # # plt.hist(all_err, bins=24, color='c', edgecolor='k', alpha=0.65)
    # # plt.axvline(all_err.mean(), color='k', linestyle='dashed', linewidth=1)
    # # plt.savefig("world_error_histogram.png")
    # # plt.show()

    # # save meta dict
    # with open(f"widowx1_c0_world_error.pkl", "wb") as f:
    #     pickle.dump(meta_dict, f)
