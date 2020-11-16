import logging
import os
import pickle
from collections import defaultdict

import colorlog
import imageio
import ipdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from src.env.fetch.fetch_push import FetchPushEnv

# from src.env.fetch.clutter_push import ClutterPushEnv
from src.prediction.losses import InpaintBlurCost, mse_criterion
from src.prediction.models.dynamics import DynamicsModel
from src.utils.plot import save_gif
from src.utils.video_recorder import VideoRecorder
from torch import cat
from torch.distributions.normal import Normal
from torchvision.transforms import ToTensor


def cem_model_planner(model: DynamicsModel, env, start, goal, cost, config):
    """
    Use learned model to test cem algorithm.
    Need the cost function, goal image
    Start is a tuple of (start img, start robot, start sim) where each is (J, _)
    Goal is a goal img
    """
    info = {}
    dev = config.device
    debug = config.debug_cem
    # Hyperparameters
    L = config.horizon  # Prediction window size
    I = config.opt_iter  # Number of optimization iterations
    J = config.action_candidates  # Number of candidate action sequences
    K = (
        config.topk
    )  # Number of top K candidate action sequences to select for optimization

    A = config.action_dim

    # Initialize action sequence belief as standard normal, of shape (L, A)
    mean = torch.zeros(L, A).to(dev)
    std = torch.ones(L, A).to(dev)
    original_env_state = env.get_state()
    ret_topks = []
    start_sim, start_robot, start_img = start
    # Optimization loop
    for i in range(I):
        model.reset(batch_size=J)
        # Sample J candidate action sequence
        m = Normal(mean, std)
        act_seq = m.sample((J,)).to(dev)  # of shape (J, L, A)
        # Generate J rollouts
        ret_preds = torch.zeros(J).to(dev)
        # duplicate the starting states J times
        curr_img = (
            (ToTensor()(start_img.copy())).expand(J, -1, -1, -1).to(dev)
        )  # (J x |I|)
        curr_robot = (
            torch.from_numpy(start_robot.copy()).expand(J, -1).to(dev)
        )  # J x |A|)
        curr_sim = [start_sim] * J  # (J x D)
        for t in range(L):
            ac = act_seq[:, t]  # (J, |A|)
            # compute the next img
            curr_img = model.next_img(curr_img, curr_robot, ac, t == 0)
            if debug and i == I - 1:
                if t == 0:
                    # curr img should be J, C, W, H
                    debug_preds = torch.zeros((L, *curr_img.shape)).to(dev)
                debug_preds[t] = curr_img
            # compute the future robot and sim using kinematics solver like mujoco
            for j in range(J):
                next_robot, next_sim = env.robot_kinematics(
                    curr_sim[j], ac[j].cpu().numpy()
                )
                curr_robot[j] = torch.from_numpy(next_robot).to(dev)
                curr_sim[j] = next_sim
                if config.reward_type == "inpaint-blur":
                    blur = t < L - config.unblur_timestep
                    rew = cost(curr_img[j], goal, blur)
                elif config.reward_type == "inpaint":
                    rew = -1 * cost(curr_img[j], goal).cpu().item()
                ret_preds[j] += rew

        # Select top K action sequences based on cumulative rewards
        ret_topk, idx = ret_preds.topk(K)
        top_act_seq = torch.index_select(
            act_seq, dim=0, index=idx
        )  # of shape (K, L, A)

        ret_topks.append("%.3f" % ret_topk.mean())  # Record mean of top returns
        # Update parameters for normal distribution
        std, mean = torch.std_mean(top_act_seq, dim=0)

    # Print means of top returns, for debugging
    # print("\tMeans of top returns: ", ret_topks)
    if config.debug_cem:
        idx = idx[:3]  # just save top 3 trajectories
        info["top_preds"] = torch.index_select(debug_preds, dim=1, index=idx).cpu()
    env.set_state(original_env_state)
    # Return first action mean, of shape (A)
    return mean[: config.replan_every, :].cpu().numpy(), info


def cem_env_planner(env, config):
    """Use actual gym environment to test cem algorithm"""
    # Hyperparameters
    L = config.horizon  # Prediction window size
    I = config.opt_iter  # Number of optimization iterations
    J = config.action_candidates  # Number of candidate action sequences
    K = (
        config.topk
    )  # Number of top K candidate action sequences to select for optimization

    # Infer action size
    A = env.action_space.shape[0]

    # Initialize action sequence belief as standard normal, of shape (L, A)
    mean = torch.zeros(L, A)
    std = torch.ones(L, A)

    ret_topks = []  # for debugging
    # Optimization loop
    for i in range(I):  # Use tqdm to track progress
        # Sample J candidate action sequence
        m = Normal(mean, std)
        act_seq = m.sample((J,))  # of shape (J, L, A)

        # Generate J rollouts
        ret_preds = torch.zeros(J)
        for j in range(J):
            # Copy environment with its state, goal, and set to dense reward
            # use set_state and get_state
            env_state = env.get_state()
            env._use_unblur = False
            for l in range(L):
                action = act_seq[j, l].numpy()
                env._use_unblur = l >= L - config.unblur_timestep
                _, rew, _, _ = env.step(action, compute_reward=True)  # Take one step
                ret_preds[j] += rew  # accumulate rewards
                env._use_unblur = False
            env.set_state(env_state)  # reset env to before rollout

        # Select top K action sequences based on cumulative rewards
        ret_topk, idx = ret_preds.topk(K)
        top_act_seq = torch.index_select(
            act_seq, dim=0, index=idx
        )  # of shape (K, L, A)
        ret_topks.append("%.3f" % ret_topk.mean())  # Record mean of top returns

        # Update parameters for normal distribution
        std, mean = torch.std_mean(top_act_seq, dim=0)

    # Print means of top returns, for debugging
    # print("\tMeans of top returns: ", ret_topks)
    return mean[: config.replan_every, :]


def run_cem_episodes(config):
    logger = colorlog.getLogger("file/console")
    num_episodes = config.num_episodes
    model = None
    use_env = config.use_env_dynamics
    env = FetchPushEnv(config)
    if not use_env:
        model = DynamicsModel(config)
        model.load_model(config.dynamics_model_ckpt)
        if config.reward_type == "inpaint":
            cost = mse_criterion
        elif config.reward_type == "inpaint-blur":
            cost = InpaintBlurCost(config)
    # Do rollouts of CEM control
    all_episode_stats = defaultdict(list)
    for i in range(num_episodes):  # this can be parallelized
        trajectory = defaultdict(list)
        obs = env.reset()
        goal = (ToTensor()(env._unblurred_goal)).to(config.device)
        if config.record_trajectory:
            trajectory["obs"].append(obs)
            trajectory["state"].append(env.get_state())
        vr = VideoRecorder(
            env,
            path=os.path.join(config.video_dir, f"test_{i}.mp4"),
            enabled=i % config.record_video_interval == 0,
            store_goal=True,
        )
        vr.capture_frame()
        terminate_ep = False
        s = 0  # Step count
        logger.info("\n=== Episode %d ===\n" % (i))
        gif = []
        while True:
            logger.info("\tStep {}".format(s))
            if use_env:
                action_seq = cem_env_planner(env, config).numpy()
            else:
                sim_state = env.get_state()
                robot = obs["robot"].astype(np.float32)
                img = obs["observation"]
                start = (sim_state, robot, img)
                action_seq, info = cem_model_planner(
                    model, env, start, goal, cost, config
                )
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

            for action in action_seq:
                obs, _, done, info = env.step(action, compute_reward=False)
                if config.record_trajectory:
                    trajectory["obs"].append(obs)
                    trajectory["ac"].append(action)
                    trajectory["state"].append(env.get_state())
                s += 1
                gif.append(obs["observation"])
                vr.capture_frame()
                dist_str = "\t"
                for k, v in info.items():
                    if "dist" in k and v > 0.01:
                        dist_str += f"{k}: {v} "
                logger.info(dist_str)
                # don't care about success as early termination
                if done or s > config.max_episode_length:
                    imageio.mimwrite("inpaintobs.gif", gif)
                    terminate_ep = True
                    logger.info("=" * 10 + f"Episode {i}" + "=" * 10)
                    if config.record_trajectory and (
                        i % config.record_trajectory_interval == 0
                    ):
                        path = os.path.join(config.trajectory_dir, f"ep_{i}.pkl")
                        with open(path, "wb") as f:
                            pickle.dump(trajectory, f)
                    # log the last step's information
                    for k, v in info.items():
                        logger.info(f"{k}: {v}")
                        all_episode_stats[k].append(v)
                    break
            if terminate_ep:
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