from argparse import Namespace

import h5py
import imageio
import matplotlib.pyplot as plt
import torch
import numpy as np

from src.dataset.locobot.sim_pick_dataset import SimPickDataset
from src.prediction.models.dynamics import SVGModel
from torch import cat
from torch.utils.data import DataLoader
from src.utils.image import zero_robot_region
from src.utils.plot import putText

def load_model(cfg: Namespace) -> SVGModel:
    model = SVGModel(cfg)
    ckpt = torch.load(cfg.dynamics_model_ckpt, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    return model

def load_demo(path: str) -> dict:
    # load start,
    demo = {}
    with h5py.File(path, "r") as hf:
        demo["obj_qpos"] = hf["obj_qpos"][:]
        demo["qpos"] = hf["qpos"][:]
        demo["eef_states"] = hf["states"][:]
        demo["observations"] = hf["observations"][:]
        demo["obj_observations"] = hf["obj_only_imgs"][:]
        demo["masks"] = hf["masks"][:]
        demo["actions"] = hf["actions"][:][:, (0,1,2,4)]
    return demo

def visualize_demo(demo_path: str ) -> None:
    demo = {}
    with h5py.File(demo_path, "r") as hf:
        demo["obj_qpos"] = hf["obj_qpos"][:]
        demo["qpos"] = hf["qpos"][:]
        demo["eef_states"] = hf["states"][:]
        demo["observations"] = hf["observations"][:]
        demo["obj_observations"] = hf["obj_only_imgs"][:]
        demo["masks"] = hf["masks"][:]
        demo["actions"] = hf["actions"][:][:, (0,1,2,4)]

    for i, obs in enumerate(demo["observations"]):
        putText(obs, f"{i+1}", (0, 8), color=(255, 255, 255))
        imageio.imwrite(f"obs_{i}.png", obs)

    imageio.imwrite(f"obj_goal_obs_{i}.png", demo["obj_observations"][-1])
    imageio.imwrite(f"robot_goal_obs_{i}.png", demo["observations"][-1])

    imageio.mimwrite("demo.gif",  demo["observations"], fps=3)

if __name__ == "__main__":
    from src.config import argparser

    cfg, _ = argparser()
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.data_root = ""

    MODEL_CKPT = "checkpoints/cost_pick_ranofuture_60000.pt"
    # DEMO_CKPT = "demos/locobot_latent_cost/none_pick_0_198_s.hdf5"
    DEMO_CKPT = "demos/fetch_pick/none_pick_0_7_f.hdf5"

    TRAIN_MODE = False # eval or train for encoder
    GOAL_IMG_TYPE = "robot" # obj, robot in goal image
    NORMALIZE_COST = True

    cfg.dynamics_model_ckpt = MODEL_CKPT
    model = load_model(cfg)
    if TRAIN_MODE:
        model.train()
        train_mode = "train"
    else:
        model.eval()
        train_mode = "eval"

    encoder = model.encoder
    visualize_demo(DEMO_CKPT)
    # prepare demo for input to the enc
    data = SimPickDataset([DEMO_CKPT], ["locobot"], cfg)

    if GOAL_IMG_TYPE == "obj":
        demo = load_demo(DEMO_CKPT)
        obj_observations = demo["obj_observations"]
        obj_masks = np.zeros_like(demo["masks"])
        # preprocess the obj images and masks
        obj_observations, obj_masks = data._preprocess_images_masks(obj_observations, obj_masks)

    train_loader = DataLoader(
        data,
        num_workers=0,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    # get a latent from each frame in the video
    with torch.no_grad():
        for x in train_loader:
            image = x["images"].squeeze(0)
            mask = x["masks"].squeeze(0)
            if GOAL_IMG_TYPE == "obj":
                image = cat((image, obj_observations[-1].unsqueeze(0)), 0)
                mask = cat((mask, obj_masks[-1].unsqueeze(0)), 0)

            image = image.to(cfg.device)
            mask = mask.to(cfg.device)

            if "dontcare" in cfg.reconstruction_loss or cfg.black_robot_input:
                image_black = zero_robot_region(mask, image)

            if cfg.model_use_mask:
                latent, _ = encoder(cat([image_black, mask], dim=1))
            else:
                latent, _ = encoder(image_black)
            break

    # plot the latent over time
    l2_dist = (latent - latent[-1]).pow(2).sum(1).sqrt()
    if GOAL_IMG_TYPE == "obj":
        l2_dist = l2_dist[:-1]
    costs = l2_dist.cpu().numpy()
    if NORMALIZE_COST:
        min = np.min(costs.min())
        max = np.max(costs.max())
        costs = (costs - min) / (max - min)
    timesteps = np.arange(len(costs)) + 1
    plt.plot(timesteps, costs, label="Latent RAC L2 dist")

    # plot the pixel cost over time
    l2_dist = (image - image[-1]).pow(2).sum((1,2,3)).sqrt()
    if GOAL_IMG_TYPE == "obj":
        l2_dist = l2_dist[:-1]
    costs = l2_dist.cpu().numpy()
    if NORMALIZE_COST:
        min = np.min(costs.min())
        max = np.max(costs.max())
        costs = (costs - min) / (max - min)
    timesteps = np.arange(len(costs)) + 1
    plt.plot(timesteps, costs, label="Pixel L2 dist")


    # plot the RA pixel cost over time
    diff = image - image[-1]
    mask = mask.type(torch.bool)
    repeat_mask = mask.repeat(1,3,1,1) # repeat channel dim
    num_world_pixels = (~repeat_mask).sum((1,2,3)) + 1
    costs = diff.pow(2).sum((1,2,3)).sqrt() / num_world_pixels
    costs = costs.cpu().numpy()
    if GOAL_IMG_TYPE == "obj":
        costs = costs[:-1]
    if NORMALIZE_COST:
        min = np.min(costs.min())
        max = np.max(costs.max())
        costs = (costs - min) / (max - min)
    timesteps = np.arange(len(costs)) + 1
    plt.plot(timesteps, costs, label="Pixel RAC L2 dist")


    # compile into a plot
    plt.xticks(timesteps)
    plt.title(f"costs w/ {train_mode} mode encoder")
    plt.legend(loc="upper right")
    plt.savefig(f"{GOAL_IMG_TYPE}_goal_costs.png")
    plt.close("all")
