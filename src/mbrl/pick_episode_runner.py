import logging
import os
import pickle
from collections import defaultdict
from os.path import join
from src.cem.pick.cem import CEMPolicy
from src.utils.mujoco import init_mjrender_device
import colorlog
import h5py
import imageio
import numpy as np
import torch
import wandb
from tqdm import trange
from src.utils.state import State, DemoGoalState
from src.env.robotics.locobot_pick_env import LocobotPickEnv
from src.prediction.losses import RobotWorldCost


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
        logger.info(f"=== Episode {ep_num}, {demo_name} ===")
        ep_stats = defaultdict(list)
        trajectory = defaultdict(list)

        demo = self._load_demo(demo_path)
        start_timestep = 2
        initial_state = {"qpos": demo["qpos"][start_timestep], "obj_qpos": demo["obj_qpos"][start_timestep]}
        goal_pos = demo["obj_qpos"][-1][:3]
        obs = env.reset(initial_state=initial_state)
        trajectory["obs"].append(obs)
        obj_dist = init_obj_dist = np.linalg.norm(obs["obj_qpos"][:3] - goal_pos)
        gripper_dist = np.linalg.norm(obs["obj_qpos"][:3] - env.get_gripper_world_pos())
        print("initial obj-goal dist", obj_dist)
        print("initial obj-gripper dist", gripper_dist)

        goal_timestep = start_timestep

        # begin policy rollout
        success = False
        ep_timestep = start_timestep
        # before_img = env._get_obs()["observation"]
        # imageio.imwrite("before_ac.png", before_img)
        while True:
            curr_img = obs["observation"]
            curr_mask = obs["masks"]
            curr_state = obs["states"]
            curr_sim_state = None
            curr_sim_state = env.get_flattened_state().copy()
            start = State(img=curr_img, state=curr_state, mask=curr_mask, sim_state=curr_sim_state)
            goal_imgs = demo["observations"][goal_timestep + 1:]
            # goal_imgs = demo["obj_observations"][goal_timestep + 1:]

            goal_masks = None
            goal_masks = demo["masks"][goal_timestep + 1:]
            # goal_masks = np.zeros_like(demo["masks"][goal_timestep + 1:])
            goal_eef_states = demo["eef_states"][goal_timestep:]
            opt_traj = demo["actions"][goal_timestep:]
            goal = DemoGoalState(imgs=goal_imgs, masks=goal_masks, states=goal_eef_states)

            ac = self.policy.get_action(start, goal, ep_num, ep_timestep, opt_traj)[0]
            # ac = np.random.uniform(low=-1, high=1, size=4)
            # ac = demo["actions"][ep_timestep, (0,1,2,4)]
            # print("ac", ac)
            # curr_eef_state = env.get_gripper_world_pos()
            # print("obj-gripper dist", np.linalg.norm(curr_eef_state - obs["obj_qpos"][:3]))
            trajectory["ac"].append(ac)
            obs, _, _, _ = env.step(ac)
            trajectory["obs"].append(obs)
            after_img = env._get_obs()["observation"]
            # imageio.imwrite("after_ac.png", after_img)
            # imageio.imwrite("after_ac_obs.png", obs["observation"])

            ep_timestep += 1
            # TODO: update goal timestep based on progress
            goal_timestep += 1
            if goal_timestep + 1 >= len(demo["observations"]):
                goal_timestep -= 1
            obj_dist = np.linalg.norm(obs["obj_qpos"][:3] - goal_pos)
            print("step", ep_timestep, obj_dist)
            success = obj_dist < 0.01
            # if object dist is larger than 3cm from the starting object dist, exit
            if (obj_dist - 0.03) >= init_obj_dist:
                print("early exiting")
                break
            if ep_timestep >= 12 or obj_dist < 0.01:
                break
        if success:
            logger.info("SUCCESS")

        record_path = join(cfg.log_dir, f"exec_{ep_num}.gif")
        if cfg.reward_type == "dontcare":
            # imageio.mimwrite(record_path, [zero_robot_region(x["masks"], x["observation"]) for x in trajectory["obs"]])
            imageio.mimwrite(record_path, [x["observation"] for x in trajectory["obs"]], fps=4)
        else:
            imageio.mimwrite(record_path, [x["observation"] for x in trajectory["obs"]], fps=4)
        record_path = join(cfg.log_dir, f"true_exec_{ep_num}.gif")
        imageio.mimwrite(record_path, demo["observations"][start_timestep:], fps=4)
        ep_stats["success"] = success
        return ep_stats

    def run(self):
        """
        Run all episodes and log their metrics
        """
        trials_per_demo = 5
        files = self._load_demo_dataset(self._config)
        all_files = []
        success = 0
        all_stats = []
        for f in files:
            all_files.extend([f for _ in range(trials_per_demo)])

        for i in trange(len(all_files), desc="running episode"):
            demo_name, demo_path = all_files[i]
            ep_stats = self.run_episode(i, demo_name, demo_path)
            all_stats.append(ep_stats)
            success += ep_stats["success"]
            print(f"running success {success}/{i+1}",  success / (i + 1))

        self._logger.info("\n\n### Summary ###")
        self._logger.info(f"Success {success}/{len(all_files)},   {(success / len(all_files)):.2f}%")

        with open(join(self._config.log_dir, "stats.pkl"), "wb") as f:
            pickle.dump(all_stats, f)

    def _load_demo_dataset(self, config):
        files = []
        for d in os.scandir(config.object_demo_dir):
            if d.is_file() and d.name.endswith("hdf5"):
                files.append((d.name, d.path))
        files.sort(key=lambda e: e[0])
        return files[: config.num_episodes]

    def _load_demo(self, path):
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

    demo_path = "/home/pallab/roboaware/demos/locobot_pick_demos/pick_0_11_s.hdf5"
    # demo_path = "/home/ed/roboaware/demos/fetch_pick/pick_0_3_f.hdf5"
    demo_name = "pick_0_11_s"
    config, _ = argparser()

    # env
    config.modified = False
    config.object_demo_dir = "/home/pallab/roboaware/demos/locobot_pick_demos"
    # config.object_demo_dir = "/home/pallab/roboaware/demos/fetch_pick_demos"
    # config.object_demo_dir = "/home/edward/roboaware/demos/fetch_pick_demos"

    # model
    config.dynamics_model_ckpt = "/home/pallab/roboaware/checkpoints/pick2_vfs_90000.pt"
    config.action_dim = 5
    # config.action_dim = 5
    # config.robot_dim = 5
    # config.model_use_robot_state = True
    # config.model_use_mask = True
    # config.model_use_future_mask = False
    # config.model_use_future_robot_state = False
    # config.lstm_group_norm = True
    # config.g_dim = 256
    # config.z_dim  = 64


    # cem
    config.debug_cem = True
    config.action_candidates = 250
    config.candidates_batch_size = 250
    config.use_env_dynamics = False
    config.cem_init_std = 0.5
    config.horizon = 5
    config.opt_iter = 5

    # cost
    config.reward_type = "dense"
    config.sparse_cost = False
    config.robot_cost_weight = 0
    config.world_cost_weight = 1

    runner = EpisodeRunner(config)
    runner.run()
    # runner.run_episode(0, demo_name, demo_path)
