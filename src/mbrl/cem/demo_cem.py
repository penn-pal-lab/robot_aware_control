import logging
import os
import pickle
from collections import defaultdict

import colorlog
import h5py
import imageio
import ipdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from src.env.fetch.clutter_push import ClutterPushEnv
from src.prediction.losses import InpaintBlurCost, mse_criterion
from src.prediction.models.dynamics import DynamicsModel
from src.utils.plot import save_gif
from src.utils.video_recorder import VideoRecorder
from torch import cat
from torch.distributions.normal import Normal
from torchvision.transforms import ToTensor
from src.mbrl.cem.cem import cem_model_planner, cem_env_planner
from torchvision.datasets.folder import has_file_allowed_extension


def load_demo_dataset(config):
    file_type = "hdf5"
    files = []
    for d in os.scandir(config.object_demo_dir):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            if config.demo_difficulty in d.name:
                files.append(d.path)
    assert config.num_episodes <= len(files), f"need at least {config.num_episodes} demos"
    return files[:config.num_episodes]

def load_demo(path):
    with h5py.File(path, "r") as hf:
        frames = hf["frames"][:]
        states = hf["states"][:]
        info = {}
        for k,v in hf.attrs.items():
            info[k] = v
    return frames, states, info


def run_cem_episodes(config):
    demos = load_demo_dataset(config)
    logger = colorlog.getLogger("file/console")
    num_episodes = config.num_episodes
    model = None
    use_env = config.use_env_dynamics
    env = ClutterPushEnv(config)
    if not use_env:
        model = DynamicsModel(config)
        model.load_model(config.dynamics_model_ckpt)
        if config.reward_type == "inpaint":
            cost = mse_criterion
        elif config.reward_type == "inpaint-blur":
            cost = InpaintBlurCost(config)
    # Do rollouts of CEM control
    all_episode_stats = defaultdict(list)
    goal_idx = 1 # start at 2nd goal since 1st is trivial
    for i in range(num_episodes):  # this can be parallelized
        frames, demo_states, demo_info = load_demo(demos[i])
        trajectory = defaultdict(list)
        goal_img = frames[goal_idx]
        goal_state = demo_states[goal_idx]
        env.reset(goal_img, goal_state, demo_info)
        # mask = np.abs(obs["observation"] - frames[0]).astype(np.uint8)
        # all_imgs = np.concatenate([frames[0], obs["observation"], mask], axis=1)
        # imageio.imwrite("comp.png", all_imgs)
        # import ipdb; ipdb.set_trace()

        goal = (ToTensor()(env._unblurred_goal)).to(config.device)
        if config.record_trajectory:
            trajectory["obs"].append(obs)
            trajectory["state"].append(env.get_state())
        vr = VideoRecorder(
            env,
            path=os.path.join(config.video_dir, f"test_{i}.mp4"),
            enabled=i % config.record_video_interval == 0,
        )

        s = 0  # Step count
        logger.info("\n=== Episode %d ===\n" % (i))
        while True:
            logger.info("\tStep {}".format(s))
            if use_env:
                action = cem_env_planner(env, config).numpy()
            else:
                sim_state = env.get_state()
                robot = obs["robot"].astype(np.float32)
                img = obs["observation"]
                start = (sim_state, robot, img)
                action, info = cem_model_planner(model, env, start, goal, cost, config)
                if config.debug_cem:
                    # (L, K, C, W, H)
                    cem_preds = info["top_preds"]
                    # create (L, 1, C, W, H) goal tensor
                    L = cem_preds.shape[0]
                    goal_ep = goal.cpu().unsqueeze(0)
                    goal_ep = goal_ep.expand(L, -1, -1, -1, -1)
                    cem_eps = cat([goal_ep, cem_preds], dim=1)
                    debug_path = os.path.join(config.plot_dir, f"ep{i}_step{s}_cem.gif")
                    save_gif(debug_path, cem_eps)

            obs, _, done, info = env.step(action, compute_reward=False)
            if config.record_trajectory:
                trajectory["obs"].append(obs)
                trajectory["ac"].append(action)
                trajectory["state"].append(env.get_state())
            s += 1
            vr.capture_frame()
            logger.info("\tObject Dist: {}".format(info["object_dist"]))
            succ = info["is_success"]
            # don't care about success as early termination
            if done or s > config.max_episode_length:
                logger.info("=" * 10 + f"Episode {i}" + "=" * 10)
                if config.record_trajectory and (
                    i % config.record_trajectory_interval == 0
                ):
                    path = os.path.join(config.trajectory_dir, f"ep_s{succ}_{i}.pkl")
                    with open(path, "wb") as f:
                        pickle.dump(trajectory, f)
                # log the last step's information
                for k, v in info.items():
                    logger.info(f"{k}: {v}")
                    all_episode_stats[k].append(v)
                break
        vr.close()

    # Close video recorder
    env.close()

    # Summary
    logger.info("\n\n### Summary ###")
    histograms = {"reward", "object_dist", "gripper_dist"}
    # upload table to wandb
    table = wandb.Table(columns=list(all_episode_stats.keys()))
    table_rows = []
    for k, v in all_episode_stats.items():
        mean = np.mean(v)
        sigma = np.std(v)
        logger.info(f"{k} avg: {mean} \u00B1 {sigma}")
        table_rows.append(f"{mean} \u00B1 {sigma}")
        # log = {f"mean/{k}": mean, f"std/{k}": sigma}
        # wandb.log(log, step=0)
        if k in histograms:  # save histogram to wandb and image
            plt.hist(v)
            plt.xlabel(k)
            plt.ylabel("Count")
            wandb.log({f"hist/{k}": wandb.Image(plt)}, step=0)
            fpath = os.path.join(config.plot_dir, f"{k}_hist.png")
            plt.savefig(fpath)
            plt.close("all")

    table.add_data(*table_rows)
    wandb.log({"Results": table}, step=0)


def setup_loggers(config):
    # make folder for exp logs
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red,bold",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    # only logs to console
    logger = colorlog.getLogger("console")
    logger.setLevel(logging.DEBUG)

    ch = colorlog.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    config.log_dir = os.path.join(config.log_dir, config.jobname)
    logger.info(f"Create log directory: {config.log_dir}")
    os.makedirs(config.log_dir, exist_ok=True)

    config.plot_dir = os.path.join(config.log_dir, "plot")
    os.makedirs(config.plot_dir, exist_ok=True)

    config.video_dir = os.path.join(config.log_dir, "video")
    os.makedirs(config.video_dir, exist_ok=True)

    config.trajectory_dir = os.path.join(config.log_dir, "trajectory")
    os.makedirs(config.trajectory_dir, exist_ok=True)

    # create the file / console logger
    filelogger = colorlog.getLogger("file/console")
    filelogger.setLevel(logging.DEBUG)
    logfile_path = os.path.join(config.log_dir, "log.txt")
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s @l%(lineno)d: %(message)s", "%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)

    filelogger.addHandler(fh)
    filelogger.addHandler(ch)

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config.device = device

    # wandb stuff
    if not config.wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    os.environ["WANDB_API_KEY"] = "24e6ba2cb3e7bced52962413c58277801d14bba0"
    exclude = ["device"]
    wandb.init(
        resume=config.jobname,
        project=config.wandb_project,
        config={k: v for k, v in config.__dict__.items() if k not in exclude},
        dir=config.log_dir,
        entity=config.wandb_entity,
    )


if __name__ == "__main__":
    from src.config import argparser

    config, _ = argparser()
    setup_loggers(config)
    run_cem_episodes(config)
