import logging
import os
import pickle
from collections import defaultdict
from os.path import join
from src.cem.pick.cem import CEMPolicy
from src.utils.mujoco import init_mjrender_device
from typing import List
import colorlog
import h5py
import imageio
import numpy as np
import torch
import wandb
from numpy.linalg import norm
from tqdm import trange
from src.utils.state import State, DemoGoalState
from src.env.robotics.locobot_pick_env import LocobotPickEnv
from src.prediction.losses import RobotWorldCost
from src.utils.plot import putText


class EpisodeRunner(object):
    """
    Run the demonstration following episodes and logs the metrics
    """

    def __init__(self, config) -> None:
        self._config = config
        self._use_env = config.use_env_dynamics
        self._setup_loggers(config)
        init_mjrender_device(config)
        self._env = LocobotPickEnv(config)
        self._setup_policy(config, self._env)
        self._setup_cost(config)
        self._stats = defaultdict(list)

    def _setup_policy(self, config, env):
        self.policy: CEMPolicy = CEMPolicy(
            config,
            physics="gt" if config.use_env_dynamics else "learned",
            init_std=config.cem_init_std,
            action_candidates=config.action_candidates,
            horizon=config.horizon,
            opt_iter=config.opt_iter
        )

    def _setup_cost(self, config):
        self.cost: RobotWorldCost = RobotWorldCost(config)

    def run_episode(self, ep_num, demo_name, demo_path):
        """
        Run one demo-following episode
        """
        cfg = self._config
        env = self._env
        logger = self._logger
        trajectory = defaultdict(list)

        demo = self._load_demo(demo_path)
        start_timestep = 0
        initial_state = {"qpos": demo["qpos"][start_timestep], "obj_qpos": demo["obj_qpos"][start_timestep]}
        goal_pos = demo["obj_qpos"][-1][:3]
        obs = env.reset(initial_state=initial_state)
        trajectory["obs"].append(obs)
        obj_dist = np.linalg.norm(obs["obj_qpos"][:3] - goal_pos)
        print("initial dist", obj_dist)

        goal_timestep = start_timestep

        # begin policy rollout
        ep_timestep = start_timestep
        while True:
            curr_img = obs["observation"]
            curr_mask = obs["masks"]
            curr_state = None
            if cfg.use_env_dynamics:
                curr_state = env.get_flattened_state()
            start = State(img=curr_img, mask=curr_mask, state=curr_state)
            goal_imgs = demo["observations"][goal_timestep + 1:]
            goal_masks = demo["masks"][goal_timestep + 1:]
            opt_traj = demo["actions"][goal_timestep:]
            goal = DemoGoalState(imgs=goal_imgs, masks=goal_masks)

            ac = self.policy.get_action(start, goal, ep_num, ep_timestep, opt_traj)[0]
            # ac = demo["actions"][ep_timestep, (0,1,2,4)]
            trajectory["ac"].append(ac)
            obs, _, _, _ = env.step(ac)
            trajectory["obs"].append(obs)

            ep_timestep += 1
            # TODO: update goal timestep based on progress
            goal_timestep += 1
            if goal_timestep + 1 >= len(demo["observations"]):
                goal_timestep -= 1
            obj_dist = np.linalg.norm(obs["obj_qpos"][:3] - goal_pos)
            print("step", ep_timestep, obj_dist)
            if ep_timestep >= 15 or obj_dist < 0.02:
                break

        imageio.mimwrite("exec.gif", [x["observation"] for x in trajectory["obs"]])
        # imageio.mimwrite("true_exec.gif", demo["observations"][start_timestep:])


    def run(self):
        """
        Run all episodes and log their metrics
        """
        files = self._load_demo_dataset(self._config)
        for i in trange(self._config.num_episodes, desc="running episode"):
            demo_name, demo_path = files[i]
            self.run_episode(i, demo_name, demo_path)

    def _load_demo(self, path):
        # load start,
        demo = {}
        with h5py.File(path, "r") as hf:
            demo["obj_qpos"] = hf["obj_qpos"][:]
            demo["qpos"] = hf["qpos"][:]
            demo["observations"] = hf["observations"][:]
            demo["masks"] = hf["masks"][:]
            demo["actions"] = hf["actions"][:][:, (0,1,2,4)]
        return demo

    def _setup_loggers(self, config):
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

        config.log_dir = join(config.log_dir, config.jobname)
        logger.info(f"Create log directory: {config.log_dir}")
        os.makedirs(config.log_dir, exist_ok=True)

        config.plot_dir = join(config.log_dir, "plot")
        os.makedirs(config.plot_dir, exist_ok=True)

        config.video_dir = join(config.log_dir, "video")
        os.makedirs(config.video_dir, exist_ok=True)

        config.trajectory_dir = join(config.log_dir, "trajectory")
        os.makedirs(config.trajectory_dir, exist_ok=True)

        # create the file / console logger
        filelogger = colorlog.getLogger("file/console")
        filelogger.setLevel(logging.DEBUG)
        logfile_path = join(config.log_dir, "log.txt")
        fh = logging.FileHandler(logfile_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s @l%(lineno)d: %(message)s", "%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        filelogger.addHandler(fh)
        filelogger.addHandler(ch)
        self._logger = colorlog.getLogger("file/console")

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

    demo_path = "/home/ed/roboaware/demos/locobot_pick/pick_0_3_s.hdf5"
    demo_name = "pick_0_3"
    config, _ = argparser()
    config.use_env_dynamics = True
    config.action_dim = 4
    config.action_candidates = 100
    config.cem_init_std = 0.5
    runner = EpisodeRunner(config)
    runner.run_episode(0, demo_name, demo_path)
