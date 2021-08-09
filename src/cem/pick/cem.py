import os

import imageio
import numpy as np
import torch
from src.cem.pick.trajectory_sampler import TrajectorySampler
from src.prediction.models.dynamics import SVGConvModel
from src.utils.plot import putText
from src.utils.state import DemoGoalState, State
from torch.distributions.normal import Normal
from tqdm import trange


class CEMPolicy(object):
    """
    Given the current state, and goal images, use CEM to find the best actions.
    Either uses ground truth physics or learned physics to do the planning.
    """

    def __init__(
        self,
        cfg,
        physics="gt",
        horizon=5,
        opt_iter=10,
        action_candidates=100,
        topk=5,
        init_std=1.0,
    ):
        # Hyperparameters
        self.horizon = horizon  # Prediction window size
        self.optimization_iter = opt_iter  # Number of optimization iterations
        self.num_actions = action_candidates  # Number of candidate action sequences
        self.K = topk
        self.init_std = init_std
        self.sparse_cost = cfg.sparse_cost  # Use cost function at end of traj

        self.action_dim = cfg.action_dim
        self.cfg = cfg


        self.traj_sampler = TrajectorySampler(cfg, physics)

        self.plot_rollouts = cfg.debug_cem
        if self.plot_rollouts:
            self.debug_cem_dir = cfg.log_dir
            os.makedirs(self.debug_cem_dir, exist_ok=True)

    def get_action(self, start: State, goal: DemoGoalState, ep_num, step, opt_traj=None):
        """
        start: data class containing start img, start robot, etc.
        goal: data class containing goal imgs, goal robot, etc.
        ep_num: used for plotting rollouts
        step: used for plotting rollouts
        opt_traj: list of expert actions to execute for debug
        Returns: a list of actions to execute in the environment
        """
        T = self.horizon
        A = self.action_dim
        N = self.num_actions

        self.ep_num = ep_num
        self.step = step
        # Initialize action sequence belief as standard normal, of shape (T-1, A)
        mean = torch.zeros(T - 1, A)
        # try setting means to opt_traj mean
        # mean = mean + opt_traj[:T-1]

        std = torch.ones(T - 1, A) * self.init_std
        # gripper range should be [-0.01, 0]
        mean[:, -1] = -0.005
        std[:, -1] = 0.005

        mean_top_costs = []  # for debugging
        # Optimization loop
        for i in trange(
            self.optimization_iter, desc="CEM: Optimizing Actions"
        ):  # Use tqdm to track progress
            # Sample N candidate action sequence
            m = Normal(mean, std)
            act_seq = m.sample((N,))  # of shape (N, T-1, A)
            # if i == 0:
            #     act_seq[-1] = 0  # always have a "do nothing" action sequence in start

            act_seq.clamp_(-1, 1)  # clamp actions
            act_seq[:,-1].clamp_(-0.01, 0)  # clamp gripper actions
            # Generate N rollouts of the N action trajectories
            if i == self.optimization_iter - 1:
                rollouts = self._get_rollouts(
                    act_seq, start, goal, opt_traj, self.plot_rollouts
                )
            else:
                rollouts = self._get_rollouts(act_seq, start, goal)

            # Select top K action sequences based on cumulative cost
            costs = torch.from_numpy(rollouts["sum_cost"])
            top_costs, top_idx = costs.topk(self.K)
            top_act_seq = torch.index_select(act_seq, dim=0, index=top_idx)
            mean_top_costs.append(f"{top_costs.mean():.3f}")

            # Update parameters for normal distribution
            std, mean = torch.std_mean(top_act_seq, dim=0)
            # clamp std to positive
            std = torch.max(0.001 * torch.ones_like(std), std)

        # Print means of top costs, for debugging
        print(
            f"\tMeans of top costs: {mean_top_costs} Opt return: {rollouts['optimal_sum_cost']:.3f}"
        )
        return mean.numpy()

    def _get_rollouts(
        self, act_seq, start: State, goal: DemoGoalState, opt_traj=None, plot=False
    ):
        """
        Return the rollouts from model
        """
        rollouts = self.traj_sampler.generate_rollouts(
            act_seq,
            start,
            goal,
            ret_obs=self.plot_rollouts,
            opt_traj=opt_traj,
            suppress_print=True,
        )
        # Plot the Top K model rollouts
        if plot:
            obs = rollouts["obs"]  # K x T x C x H x W
            if opt_traj is not None:
                opt_obs = np.expand_dims(rollouts["optimal_obs"], 0)
                obs = np.concatenate([opt_obs, obs])
            obs = np.uint8(obs)
            # obs = np.uint8(255 * obs)
            # obs = obs.transpose((0, 1, 3, 4, 2))  # K x T x H x W x C
            # topk_act = act_seq[rollouts["topk_idx"]]
            gif_folder = self.debug_cem_dir
            os.makedirs(gif_folder, exist_ok=True)

            goal_img = goal.imgs[0]
            curr_img = start.img.copy()
            info_img = np.zeros_like(goal_img)
            img = np.concatenate([info_img, curr_img, goal_img], axis=1)
            putText(img, f"Start", (0, 8), color=(255, 255, 255))
            gif = [np.concatenate([img] * obs.shape[0])]
            for t in range(obs.shape[1]):
                all_k_img = []
                for k in range(obs.shape[0]):
                    curr_img = obs[k, t]
                    g = t if t < len(goal.imgs) else -1
                    goal_img = goal.imgs[g]
                    info_img = np.zeros_like(goal_img)
                    img = np.concatenate([info_img, curr_img, goal_img], axis=1).copy()
                    if opt_traj is not None:
                        if k == 0:
                            putText(img, f"Opt", (0, 8), color=(255, 255, 255))
                            # ac = opt_traj[t]
                        else:
                            putText(img, f"Rank {k-1}", (0, 8), color=(255, 255, 255))
                            # ac = topk_act[k - 1, t]
                    else:
                        putText(img, f"Rank {k}", (0, 8), color=(255, 255, 255))
                        # ac = topk_act[k, t]

                    # putText(
                    #     img, f"X:{ac[0] * 100:.1f}cm", (0, 16), color=(255, 255, 255)
                    # )
                    # putText(
                    #     img, f"Y:{ac[1] * 100:.1f}cm", (0, 24), color=(255, 255, 255)
                    # )
                    putText(img, f"{t}", (64, 8), color=(255, 255, 255))
                    putText(img, "GOAL", (128, 8), color=(255, 255, 255))
                    all_k_img.append(img)
                all_k_img = np.concatenate(all_k_img)
                gif.append(all_k_img)

            gif_path = os.path.join(gif_folder, f"step_{self.step}_top_k.gif")
            imageio.mimwrite(gif_path, gif, fps=2)
            # import ipdb; ipdb.set_trace()
        return rollouts


if __name__ == "__main__":
    """test the cem rollout
    python -m src.cem.cem --data_root ~/Robonet   --n_future 5 --n_past 1 --n_eval 10 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --checkpoint_interval 10 --eval_interval 5 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment train_locobot_singleview --preprocess_action raw --random_snippet True --model_use_mask False --model_use_robot_state False --model_use_heatmap False    --dynamics_model_ckpt locobot_689_tile_ckpt_136500.pt --debug_cem True  --candidates_batch_size 10 --action_candidates 100  --horizon 4 --opt_iter 5 --cem_init_std 0.025  --sparse_cost False --seed 0
    """
    import h5py
    import torchvision.transforms as tf
    from src.config import argparser
    from src.dataset.locobot.locobot_singleview_dataloader import create_transfer_loader
    from src.dataset.robonet.robonet_dataset import normalize

    config, _ = argparser()
    config.device = (
        torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    )
    config.batch_size = 1
    config.robot_joint_dim = 5
    config.cem_init_std = 0.015

    traj_path = None
    if traj_path is None:
        loader = create_transfer_loader(config)
        for data in loader:
            imgs = data["images"]
            states = data["states"].squeeze()
            qposes = data["qpos"].squeeze()
            print(data["file_path"])
            break
    else:
        w, h = config.image_width, config.image_height
        img_transform = tf.Compose([tf.ToTensor(), tf.Resize((h, w))])
        with h5py.File(traj_path, "r") as hf:
            IMAGE_KEY = "frames"
            if "observations" in hf:
                IMAGE_KEY = "observations"
            imgs = hf[IMAGE_KEY][:]  # already uint8 (H,W,3) numpy array
            states = hf["states"][:]
            qposes = hf["qpos"][:]
        imgs = torch.stack([img_transform(i) for i in imgs])
        imgs.unsqueeze_(0)
        # normalize the states for model
        low = torch.from_numpy(np.array([0.015, -0.3, 0.1, 0, 0], dtype=np.float32))
        high = torch.from_numpy(np.array([0.55, 0.3, 0.4, 1, 1], dtype=np.float32))
        states = normalize(states, low, high)

        qposes = torch.from_numpy(qposes)
        states = torch.from_numpy(states)

    imgs = (255 * imgs).permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
    imgs = imgs[0]

    # now choose a start and goal img from the video
    imageio.mimwrite("full_traj.gif", imgs)
    start_idx = 0
    goal_idx = 6
    # start, goal imgs are numpy arrays of (H, W, 3) uint8
    curr_img = imgs[start_idx]
    curr_state = State(curr_img, state=states[start_idx], qpos=qposes[start_idx])
    goal_imgs = [imgs[goal_idx]]
    goal_qposes = [qposes[goal_idx]]
    imageio.imwrite("start_goal.png", np.concatenate([curr_img, goal_imgs[0]], 1))
    goal_state = DemoGoalState(goal_imgs, qposes=goal_qposes)

    model = SVGConvModel(config)
    ckpt = torch.load(config.dynamics_model_ckpt, map_location=config.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    policy = CEMPolicy(config, model, init_std=0.015)
    actions = policy.get_action(curr_state, goal_state, 0, 0)
