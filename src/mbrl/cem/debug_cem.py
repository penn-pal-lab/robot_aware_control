import pickle

import numpy as np
import torch
from src.env.fetch.fetch_push import FetchPushEnv
from src.mbrl.cem.cem import DynamicsModel
from src.utils.plot import save_gif
from torchvision.transforms import ToTensor


def debug_cem(config):
    # load model for predicting dynamics
    model = DynamicsModel(config)
    model.load_model(config.dynamics_model_ckpt)
    # load optimal trajectory
    with open(config.debug_trajectory_path, "rb") as f:
        trajectory = pickle.load(f)

    env = FetchPushEnv(config)
    _ = env.reset()
    states = trajectory["state"]
    acs = trajectory["ac"]
    obs = trajectory["obs"]
    model.reset(batch_size=1)
    all_imgs = []

    start = 4
    # env.set_state(states[0])
    # rollout optimal action sequence with learned model, and see outputs
    ob = ToTensor()(obs[start]["observation"]).unsqueeze(0)
    # robot = torch.from_numpy(obs[0]["robot"].astype(np.float32)).unsqueeze(0)
    # state = states[start]
    for i in range(start + 1, len(obs)):
        # TODO: check env generated next robot state with trajectory next robot state
        ac = torch.from_numpy(acs[i-1]).unsqueeze(0)
        robot = torch.from_numpy(obs[i-1]["robot"].astype(np.float32)).unsqueeze(0)
        pred_img = model.next_img(ob, robot, ac, i == (start + 1))
        # robot, state = env.robot_kinematics(state, acs[i-1])
        # robot = torch.from_numpy(robot.astype(np.float32)).unsqueeze(0)
        next_img = ToTensor()(obs[i]["observation"]).unsqueeze(0)
        mask = torch.abs(next_img - pred_img).type(torch.float32)
        all_imgs.append([pred_img.squeeze(), next_img.squeeze(), mask.squeeze()])
        ob = pred_img
    save_gif(f"{start}.gif", all_imgs)




if __name__ == "__main__":
    from src.config import argparser
    config, _ = argparser()
    debug_cem(config)
