import logging
import os
import pickle
from collections import defaultdict

import colorlog
import h5py
import imageio
import ipdb
import matplotlib
import numpy as np
import torch
import wandb
from src.env.fetch.clutter_push import ClutterPushEnv
from src.prediction.losses import InpaintBlurCost
from src.prediction.models.dynamics import DynamicsModel
from src.cem.trajectory_sampler import generate_env_rollouts
from src.utils.plot import putText
from torch import cat
from torch.distributions.normal import Normal
from torchvision.transforms import ToTensor
from torchvision.datasets.folder import has_file_allowed_extension


def cem_env_planner(env, goal_imgs, cost, cfg):
    """
    Use actual physics engine as dynamics model for CEM
    """
    # Hyperparameters
    L = cfg.horizon  # Prediction window size
    I = cfg.opt_iter  # Number of optimization iterations
    J = cfg.action_candidates  # Number of candidate action sequences
    K = cfg.topk

    # Infer action size
    A = env.action_space.shape[0]

    # Initialize action sequence belief as standard normal, of shape (L, A)
    mean = torch.zeros(L, A)
    std = torch.ones(L, A)
    mean_top_costs = []  # for debugging
    # Optimization loop
    for i in range(I):  # Use tqdm to track progress
        # Sample J candidate action sequence
        m = Normal(mean, std)
        act_seq = m.sample((J,))  # of shape (J, L, A)
        # Generate J rollouts
        rollouts = generate_env_rollouts(cfg, env, act_seq, goal_imgs)

        # Select top K action sequences based on cumulative cost
        costs = torch.from_numpy(rollouts["sum_cost"])
        top_costs, top_idx = costs.topk(K)
        top_act_seq = torch.index_select(act_seq, dim=0, index=top_idx)
        mean_top_costs.append(f"{top_costs.mean():.3f}")

        # Update parameters for normal distribution
        std, mean = torch.std_mean(top_act_seq, dim=0)

    # Print means of top costs, for debugging
    print(
        f"\tMeans of top costs: {mean_top_costs} Opt return: {rollouts['optimal_sum_cost']:.3f}"
    )
    # save gifs of top trajectories for debugging
    # Return first action mean, of shape (A)
    return mean[: cfg.replan_every, :]


def load_demo_dataset(config):
    file_type = "hdf5"
    files = []
    for d in os.scandir(config.object_demo_dir):
        if d.is_file() and has_file_allowed_extension(d.path, file_type):
            files.append((d.name, d.path))
    assert config.num_episodes <= len(
        files
    ), f"need at least {config.num_episodes} demos"
    return files[: config.num_episodes]


def load_object_demo(path):
    # contains the object only image sequence, and poses of each object
    demo = {}
    with h5py.File(path, "r") as hf:
        demo["pushed_obj"] = hf.attrs["pushed_obj"]
        demo["states"] = hf["states"][:]
        # demo["robot_demo"] = hf["robot_demo"][:]
        for k, v in hf.items():
            if "object" in k:
                demo[k] = v[:]
    return demo


def run_cem_episodes(config):
    logger = colorlog.getLogger("file/console")
    files = load_demo_dataset(config)
    num_episodes = config.num_episodes
    model = None
    use_env = config.use_env_dynamics
    timescale = config.demo_timescale
    env = ClutterPushEnv(config)
    if not use_env:
        model = DynamicsModel(config)
        model.load_model(config.dynamics_model_ckpt)
    if config.reward_type == "inpaint-blur":
        cost = InpaintBlurCost(config)
    else:
        cost = lambda a, b: -np.linalg.norm(a - b)
    # Do rollouts of CEM control
    all_episode_stats = defaultdict(list)
    for i in range(num_episodes):  # this can be parallelized
        demo_name, demo_path = files[i]
        demo = load_object_demo(demo_path)
        # use for debugging
        optimal_traj = demo["object_inpaint_demo"][::timescale]
        demo_goal_imgs = demo["object_only_demo"][::timescale]
        pushed_obj = demo["pushed_obj"] + ":joint"
        goal_obj_poses = demo[pushed_obj][::timescale]
        push_length = np.linalg.norm(goal_obj_poses[-1][:2] - goal_obj_poses[0][:2])

        subgoal_idx = config.subgoal_start
        goal_imgs = demo_goal_imgs[subgoal_idx:]
        goal_img = goal_imgs[0]

        trajectory = defaultdict(list)
        init_state = demo["states"][0]
        terminate_ep = False
        obs = env.reset(init_state)
        if config.record_trajectory:
            trajectory["obs"].append(obs)
            trajectory["state"].append(env.get_state())

        record = i % config.record_video_interval == 0
        s = 0  # Step count
        if record:
            gif = []
            env_ob = env.render("rgb_array")  # no inpainting
            ob = obs["observation"]  # inpainting
            gif_img = np.concatenate([env_ob, ob, goal_img], axis=1)
            rew = cost(ob, goal_img)
            goal_str = f"{subgoal_idx}/{len(demo_goal_imgs)-1}"
            time_str = f"{s}/{config.max_episode_length}"
            gif_img = create_gif_img(env_ob, ob, goal_img, time_str, rew, goal_str)
            gif.append(gif_img)

        logger.info(f"=== Episode {i}, {demo_name} ===")
        logger.info(f"Pushing {demo['pushed_obj']} for {(push_length * 100):.1f} cm\n")
        while True:
            logger.info(f"\tStep {s + 1}")
            goal_imgs = demo_goal_imgs[subgoal_idx:]
            goal_img = goal_imgs[0]
            if config.demo_cost:
                config.optimal_traj = optimal_traj[subgoal_idx:]
            if use_env:
                actions = cem_env_planner(env, goal_imgs, cost, config).numpy()
            for action in actions:
                obs, _, _, _ = env.step(action)
                curr_img = obs["observation"]
                rew = cost(curr_img, goal_img)
                curr_obj_pos = obs[pushed_obj][:2]
                goal_obj_pos = goal_obj_poses[subgoal_idx][:2]
                final_goal_obj_pos = goal_obj_poses[-1][:2]
                obj_dist = np.linalg.norm(curr_obj_pos - goal_obj_pos)
                final_obj_dist = np.linalg.norm(curr_obj_pos - final_goal_obj_pos)
                print(
                    f"Current goal: {subgoal_idx}/{len(demo_goal_imgs)-1}, dist to goal: {obj_dist}, dist to last goal: { final_obj_dist}"
                )
                print("Reward:", rew)
                if config.record_trajectory:
                    trajectory["obs"].append(obs)
                    trajectory["ac"].append(action)
                    trajectory["state"].append(env.get_state())
                s += 1
                if record:
                    env_ob = env.render("rgb_array")  # no inpainting
                    ob = obs["observation"]  # inpainting
                    goal_str = f"{subgoal_idx}/{len(demo_goal_imgs)-1}"
                    time_str = f"{s}/{config.max_episode_length}"
                    gif_img = create_gif_img(
                        env_ob, ob, goal_img, time_str, rew, goal_str
                    )
                    gif.append(gif_img)

                # set the most future subgoal that is still <= threshold, and start from there
                finish_demo = False
                if config.sequential_subgoal:
                    if np.linalg.norm(curr_img - goal_img) < config.subgoal_threshold:
                        subgoal_idx += 1
                        finish_demo = subgoal_idx >= len(demo_goal_imgs)
                else:
                    all_goal_diffs = curr_img - demo_goal_imgs[subgoal_idx:]
                    min_idx = 0
                    new_subgoal = False
                    for j, goal_diff in enumerate(all_goal_diffs):
                        goal_cost = np.linalg.norm(goal_diff)
                        if goal_cost <= config.subgoal_threshold:
                            new_subgoal = True
                            min_idx = j
                    subgoal_idx += min_idx
                    if new_subgoal:
                        subgoal_idx += 1
                        finish_demo = subgoal_idx >= len(demo_goal_imgs)

                if finish_demo or s >= config.max_episode_length - 1:
                    logger.info("=" * 10 + f"Episode {i}" + "=" * 10)
                    if config.record_trajectory and (
                        i % config.record_trajectory_interval == 0
                    ):
                        path = os.path.join(
                            config.trajectory_dir, f"ep_s{subgoal_idx}_{i}.pkl"
                        )
                        with open(path, "wb") as f:
                            pickle.dump(trajectory, f)
                    subgoal_progress = (subgoal_idx - config.subgoal_start) / (
                        len(demo_goal_imgs) - config.subgoal_start
                    )
                    all_episode_stats["subgoal_progress"].append(subgoal_progress)
                    push_progress = (push_length - final_obj_dist) / push_length
                    all_episode_stats["push_progress"].append(push_progress)
                    all_episode_stats["final_obj_dist"].append(final_obj_dist)
                    terminate_ep = True
                    break

            if terminate_ep:
                break
        if record:
            gif_path = os.path.join(
                config.video_dir, f"ep_{i}_{'s' if finish_demo else 'f'}.gif"
            )
            imageio.mimwrite(gif_path, gif)

    env.close()

    # Summary
    logger.info("\n\n### Summary ###")
    # histograms = {"reward", "object_dist", "gripper_dist"}
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
        # if k in histograms:  # save histogram to wandb and image
        #     plt.hist(v)
        #     plt.xlabel(k)
        #     plt.ylabel("Count")
        #     wandb.log({f"hist/{k}": wandb.Image(plt)}, step=0)
        #     fpath = os.path.join(config.plot_dir, f"{k}_hist.png")
        #     plt.savefig(fpath)
        #     plt.close("all")

    table.add_data(*table_rows)
    wandb.log({"Results": table}, step=0)


def create_gif_img(env_ob, ob, goal, time_str, cost, goal_str):
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
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    config, _ = argparser()
    setup_loggers(config)
    run_cem_episodes(config)
