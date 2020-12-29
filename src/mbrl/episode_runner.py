import logging
import os
import pickle
from collections import defaultdict
from os.path import join
from typing import List

import colorlog
import h5py
import imageio
import numpy as np
import torch
import wandb
from numpy.linalg import norm
from tqdm import trange
from src.cem.demo_cem import DemoCEMPolicy
from src.utils.state import State, DemoGoalState
from src.env.fetch.clutter_push import ClutterPushEnv
from src.prediction.losses import RobotWorldCost
from src.utils.plot import putText
from torchvision.datasets.folder import has_file_allowed_extension


class EpisodeRunner(object):
    """
    Run the demonstration following episodes and logs the metrics
    """

    def __init__(self, config) -> None:
        self._config = config
        self._use_env = config.use_env_dynamics
        self._setup_loggers(config)
        self._env = ClutterPushEnv(config)
        self._setup_policy(config, self._env)
        self._setup_cost(config)
        self._stats = defaultdict(list)

    def _setup_policy(self, config, env):
        self.policy = DemoCEMPolicy(config, env)

    def _setup_cost(self, config):
        self.cost: RobotWorldCost = RobotWorldCost(config)

    def _pick_next_goal(self, curr: State, goal: State):
        """
        This function checks if the costs are under the threshold
        If it is, then we will pick the next goal index.
        """
        # goal choosing logic
        cfg = self._config
        if cfg.sequential_subgoal:
            robot_success = True
            if cfg.robot_cost_weight != 0:
                eef_dist = self.cost.robot_cost(curr, goal)
                robot_success = eef_dist < cfg.robot_cost_success

            world_success = True
            if cfg.world_cost_weight != 0:
                img_dist = self.cost.world_cost(curr, goal)
                world_success = img_dist < cfg.world_cost_success

            all_success = robot_success and world_success
            if all_success:
                self._g_i += 1
        else:
            raise NotImplementedError("Need to implement skip subgoal with State")
            # skip to most future goal that is still <= threshold, and start from there
            all_goal_diffs = curr_img.astype(np.float) - self.demo_goal_imgs[
                self._g_i :
            ].astype(np.float)
            min_idx = 0
            new_goal = False
            for j, goal_diff in enumerate(all_goal_diffs):
                goal_cost = norm(goal_diff)
                if goal_cost <= cfg.world_cost_success:
                    new_goal = True
                    min_idx = j + 1
            self._g_i += min_idx
            if new_goal:
                self._g_i += 1

    def run_episode(self, ep_num, demo_name, demo_path):
        """
        Run one demo-following episode
        """
        cfg = self._config
        env = self._env
        logger = self._logger
        demo = self._load_demo(demo_path)
        self.demo_goal_imgs = demo[cfg.demo_type][:: cfg.demo_timescale]
        self.demo_goal_robots = demo["robot_state"][:: cfg.demo_timescale]
        demo_goal_masks = demo["masks"][:: cfg.demo_timescale]
        # use opt trajectory for debugging
        demo_opt_traj: List[State] = []
        demo_opt_imgs = demo["object_inpaint_demo"][:: cfg.demo_timescale]
        for i in range(len(demo_opt_imgs)):
            goal_img = demo_opt_imgs[i]
            goal_robot = self.demo_goal_robots[i]
            goal_mask = demo_goal_masks[i]
            g = State(goal_img, goal_robot, goal_mask)
            demo_opt_traj.append(g)

        num_goals = len(self.demo_goal_imgs)
        pushed_obj = demo["pushed_obj"] + ":joint"
        goal_obj_poses = demo[pushed_obj][:: cfg.demo_timescale]
        push_length = norm(goal_obj_poses[-1][:2] - goal_obj_poses[0][:2])

        self._g_i = cfg.subgoal_start  # goal index
        goal_imgs = self.demo_goal_imgs[self._g_i :]
        goal_img = goal_imgs[0]
        goal_robots = self.demo_goal_robots[self._g_i :]
        goal_robot = goal_robots[0]
        goal_masks = demo_goal_masks[self._g_i :]
        goal_mask = goal_masks[0]

        trajectory = defaultdict(list)
        terminate_ep = False
        obs = env.reset(demo["states"][0])
        curr_img = obs["observation"]
        curr_robot = obs["robot"]
        curr_sim = obs["state"]
        curr_mask = obs["mask"]

        # Check if demo actions are meaningful
        if cfg.debug_cem:
            curr_state = State(curr_img, curr_robot, curr_mask, curr_sim)
            goal_state = DemoGoalState(goal_imgs, goal_robots)
            self.policy.compare_optimal_actions(demo, curr_state, goal_state, demo_name)

        if cfg.record_trajectory:
            trajectory["obs"].append(obs)
            trajectory["state"].append(env.get_state())

        self._record = ep_num % cfg.record_video_interval == 0
        self._step = 0  # Step count
        gif = []
        self._add_img_to_gif(gif, curr_img, goal_img)

        logger.info(f"=== Episode {ep_num}, {demo_name} ===")
        logger.info(f"Pushing {demo['pushed_obj']} for {(push_length * 100):.1f} cm\n")
        while True:
            logger.info(f"\tStep {self._step + 1}")
            goal_imgs = self.demo_goal_imgs[self._g_i :]
            goal_img = goal_imgs[0]
            goal_robots = self.demo_goal_robots[self._g_i :]
            goal_robot = goal_robots[0]
            opt_traj = demo_opt_traj[self._g_i :] if cfg.demo_cost else None

            # Use CEM to find the best action(s)
            curr_state = State(curr_img, curr_robot, curr_mask, curr_sim)
            goal_state = DemoGoalState(goal_imgs, goal_robots, goal_masks)
            actions = self.policy.get_action(
                curr_state, goal_state, ep_num, self._step, opt_traj=opt_traj
            )
            # See how CEM actions rollout in actual environment
            # if cfg.debug_cem:
            #     gif_folder = join(cfg.log_dir, "debug_cem")
            #     os.makedirs(f"{gif_folder}/ep_num", exist_ok=True)
            #     gif_path = f"ep_{ep_num}/step_{self._step}_env_comparison.gif"
            #     fake_demo = {"states": [curr_sim], "actions": actions}
            #     self.policy.compare_optimal_actions(
            #         fake_demo, curr_state, goal_state, gif_path
            #     )
            actions = actions[: cfg.replan_every]
            # Execute the planned actions. Usually only 1 action
            for action in actions:
                obs, _, _, _ = env.step(action)
                curr_img = obs["observation"]
                curr_robot = obs["robot"]
                curr_sim = obs["state"]
                curr_mask = obs["mask"]

                state = State(curr_img, curr_robot, curr_mask, curr_sim)
                g_state = State(goal_img, goal_robot, goal_mask)

                rew = self.cost(state, g_state, print_cost=True)
                curr_obj_pos = obs[pushed_obj][:2]
                goal_obj_pos = goal_obj_poses[self._g_i][:2]
                final_goal_obj_pos = goal_obj_poses[-1][:2]
                obj_dist = norm(curr_obj_pos - goal_obj_pos)
                final_obj_dist = norm(curr_obj_pos - final_goal_obj_pos)
                logger.info(
                    f"Current goal: {self._g_i}/{num_goals-1}, dist to goal: {obj_dist:.4f}, dist to last goal: {final_obj_dist:.4f}"
                )
                logger.info(f"Reward:{rew:.2f}")
                if cfg.record_trajectory:
                    trajectory["obs"].append(obs)
                    trajectory["ac"].append(action)
                    trajectory["state"].append(env.get_state())
                self._step += 1
                self._add_img_to_gif(gif, curr_img, goal_img)

                # goal choosing logic
                self._pick_next_goal(state, g_state)
                finish_demo = self._g_i >= num_goals

                # if episode is done, log statistics and break out of loop
                if finish_demo or self._step >= cfg.max_episode_length - 1:
                    logger.info("=" * 10 + f"Episode {ep_num}" + "=" * 10)
                    if cfg.record_trajectory and (
                        ep_num % cfg.record_trajectory_interval == 0
                    ):
                        path = join(cfg.trajectory_dir, f"ep_s{self._g_i}_{ep_num}.pkl")
                        with open(path, "wb") as f:
                            pickle.dump(trajectory, f)
                    goal_progress = (self._g_i - cfg.subgoal_start) / (
                        num_goals - cfg.subgoal_start
                    )
                    self._stats["goal_progress"].append(goal_progress)
                    push_progress = (push_length - final_obj_dist) / push_length
                    self._stats["push_progress"].append(push_progress)
                    self._stats["final_obj_dist"].append(final_obj_dist)
                    self._stats["demo_name"].append(demo_name)
                    terminate_ep = True
                    break

            if terminate_ep:
                break
        if self._record:
            gif_path = join(
                cfg.video_dir, f"ep_{ep_num}_{'s' if finish_demo else 'f'}.gif"
            )
            imageio.mimwrite(gif_path, gif)

    def run(self):
        """
        Run all episodes and log their metrics
        """
        files = self._load_demo_dataset(self._config)
        for i in trange(self._config.num_episodes, desc="running episode"):
            demo_name, demo_path = files[i]
            self.run_episode(i, demo_name, demo_path)
        self._env.close()
        self._logger.info("\n\n### Summary ###")
        # histograms = {"reward", "object_dist", "gripper_dist"}
        # upload table to wandb
        table = wandb.Table(columns=list(self._stats.keys()))
        table_rows = []
        for k, v in self._stats.items():
            if k in ["demo_name"]:
                table_rows.append(v)
                continue
            mean = np.mean(v)
            sigma = np.std(v)
            self._logger.info(f"{k} avg: {mean} \u00B1 {sigma}")
            table_rows.append(f"{mean} \u00B1 {sigma}")
            log = {f"mean/{k}": mean, f"std/{k}": sigma}
            wandb.log(log, step=0)
            # if k in histograms:  # save histogram to wandb and image
            #     plt.hist(v)
            #     plt.xlabel(k)
            #     plt.ylabel("Count")
            #     wandb.log({f"hist/{k}": wandb.Image(plt)}, step=0)
            #     fpath = join(config.plot_dir, f"{k}_hist.png")
            #     plt.savefig(fpath)
            #     plt.close("all")

        table.add_data(*table_rows)
        wandb.log({"Results": table}, step=0)

        with open(join(config.plot_dir, "stats.pkl"), "wb") as f:
            pickle.dump(self._stats, f)

    def _load_demo_dataset(self, config):
        file_type = "hdf5"
        files = []
        for d in os.scandir(config.object_demo_dir):
            if d.is_file() and has_file_allowed_extension(d.path, file_type):
                files.append((d.name, d.path))
        assert config.num_episodes <= len(
            files
        ), f"need at least {config.num_episodes} demos"
        # sort files for comparing across runs
        files.sort(key=lambda e: e[0])
        return files[: config.num_episodes]

    def _load_demo(self, path):
        # contains the object only image sequence, and poses of each object
        demo = {}
        with h5py.File(path, "r") as hf:
            demo["pushed_obj"] = hf.attrs["pushed_obj"]
            demo["actions"] = hf["actions"][:]
            demo["states"] = hf["states"][:]
            if "masks" in hf:
                demo["masks"] = hf["masks"][:]
            if self._config.reward_type == "dontcare" and self._config.demo_type == "object_only_demo":
                demo["masks"] = np.zeros(hf["masks"].shape, dtype=np.bool)
            # import ipdb; ipdb.set_trace()
            # demo[self._config.demo_type] = hf[self._config.demo_type]
            demo["robot_demo"] = hf["robot_demo"][:]
            demo["robot_state"] = hf["robot_state"][:]
            for k, v in hf.items():
                if "object" in k:
                    demo[k] = v[:]
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

    def _add_img_to_gif(self, gif, curr_img, goal_img):
        if self._record:
            env_ob = self._env.render("rgb_array")  # no inpainting
            # rew = self.cost(curr_img, goal_img)
            rew = 0
            goal_str = f"{self._g_i}/{len(self.demo_goal_imgs)-1}"
            time_str = f"{self._step}/{config.max_episode_length}"
            gif_img = self._create_gif_img(
                env_ob, curr_img, goal_img, time_str, rew, goal_str
            )
            gif.append(gif_img)

    def _create_gif_img(self, env_ob, ob, goal, time_str, cost, goal_str):
        env_ob = env_ob.copy()
        putText(env_ob, f"REAL", (0, 8))
        putText(env_ob, time_str, (0, 126))

        ob = ob.copy()
        putText(ob, f"INPAINT", (0, 8))
        putText(ob, f"{cost:.0f}", (0, 126))

        goal = goal.copy()
        putText(goal, f"GOAL", (0, 8))
        putText(goal, goal_str, (24, 126))

        img = np.concatenate([env_ob, ob, goal], axis=1)
        return img


if __name__ == "__main__":
    from src.config import argparser

    config, _ = argparser()
    runner = EpisodeRunner(config)
    runner.run()
