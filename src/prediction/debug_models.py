"""Compares the outputs of models"""
import os
from src.prediction.losses import dontcare_mse_criterion, world_mse_criterion

import h5py
import imageio
import random
import numpy as np
import torch
from src.prediction.models.dynamics import DynamicsModel
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.transforms import ToTensor


def load_dynamics_model(config, path):
    model = DynamicsModel(config)
    model.load_model(path)
    return model


def load_visualization_data(config, num_files):
    file_type = "hdf5"
    files = [
        d.path
        for d in os.scandir(config.data_root)
        if d.is_file() and has_file_allowed_extension(d.path, file_type)
    ]
    random.seed(0)
    random.shuffle(files)
    files = files[: num_files]
    data = []
    for file_path in files:
        with h5py.File(file_path, "r") as hf:
            frames = hf["robot_demo"][:]
            robots = hf["robot_state"][:]
            actions = hf["actions"][:]
            masks = hf["masks"][:]
            data.append((frames, robots, actions, masks))
    return data
            
def tensor_to_img(tensor):
    img = (tensor.squeeze() * 255).cpu().permute(1,2,0)
    return img

@torch.no_grad()
def visualize_models(config, path_1, path_2):
    data = load_visualization_data(config, 10)
    use_cuda = torch.cuda.is_available()
    d = torch.device("cuda" if use_cuda else "cpu")
    m1 = load_dynamics_model(config, path_1)
    m2 = load_dynamics_model(config, path_2)

    m1_world_mse = m2_world_mse = 0
    num_frames = 0
    for j, (frames, robots, actions, masks) in enumerate(data):
        # get all next frame predictions
        m1.reset(1)
        m2.reset(1)
        gif = []
        for i in range(1, len(frames)):
            ob = ToTensor()(frames[i-1]).to(d)
            ac = torch.from_numpy(actions[i-1].astype(np.float32)).unsqueeze(0).to(d)
            robot = torch.from_numpy(robots[i-1].astype(np.float32)).unsqueeze(0).to(d)
            mask = torch.from_numpy(masks[i-1]).unsqueeze(0).to(d)
            ob = torch.cat([ob, mask], dim=0).unsqueeze(0).to(d)
            m1_pred = m1.next_img(ob, robot, ac, True)
            m2_pred = m2.next_img(ob, robot, ac, True)
            # get only world region
            m1_img = tensor_to_img(m1_pred)
            m2_img = tensor_to_img(m2_pred)
            future_mask = masks[i]
            # black out the robot regions
            m1_img[future_mask] = 0
            m2_img[future_mask] = 0
            # visualize them side by side, and their difference
            future_ob = ToTensor()(frames[i]).to(d)
            future_mask = torch.from_numpy(future_mask).to(d)
            m1_world_mse += world_mse_criterion(m1_pred, future_ob, future_mask)
            m2_world_mse += world_mse_criterion(m2_pred, future_ob, future_mask)
            num_frames += 1
            future_img = frames[i].copy()
            future_img[future_mask.cpu().numpy()] = 0
            # m1_diff = np.clip(np.abs(m1_img - future_img), 0, 255)
            # m2_diff = np.clip(np.abs(m2_img - future_img), 0, 255)
            m1_m2_diff = np.clip(np.abs(m1_img - m2_img), 0, 255)
            img = np.concatenate([m1_img, m2_img, m1_m2_diff], axis=1)
            gif.append(img)

        imageio.mimwrite(f"debug_{j}.gif", gif)
    m1_world_mse /= num_frames
    m2_world_mse /= num_frames
    print(f"m1_world_mse, {m1_world_mse}")
    print(f"m2_world_mse, {m2_world_mse}")
    print(f"m1-m2 diff, {m1_world_mse - m2_world_mse}")





if __name__ == "__main__":
    from src.config import argparser
    path_1 = "dontcare_prediction_128125.pt"
    path_2 = "mse_prediction_128125.pt"
    config, _ = argparser()
    visualize_models(config, path_1, path_2)
