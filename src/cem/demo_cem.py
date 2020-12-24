from src.env.fetch.clutter_push import ClutterPushEnv
import imageio
import numpy as np
import torch
from src.utils.state import State, DemoGoalState
from src.cem.trajectory_sampler import (
    generate_env_rollouts,
    generate_model_rollouts,
)
from src.prediction.models.dynamics import DynamicsModel
from src.utils.plot import putText
from torch.distributions.normal import Normal
import os


class DemoCEMPolicy(object):
    """
    Given the current state, and goal images, use CEM to find the best actions.
    Either uses ground truth physics or learned physics to do the planning.
    """

    def __init__(self, cfg, env) -> None:
        # Hyperparameters
        self.L = cfg.horizon  # Prediction window size
        self.I = cfg.opt_iter  # Number of optimization iterations
        self.J = cfg.action_candidates  # Number of candidate action sequences
        self.K = cfg.topk
        self.R = cfg.replan_every  # Number of real world actions to take
        self.std = cfg.cem_init_std
        self.sparse_cost = cfg.sparse_cost  # Use cost function at end of traj
        self.cem_init_std = cfg.cem_init_std

        # Infer action size
        self.A = env.action_space.shape[0]
        self.env: ClutterPushEnv = env
        self.cfg = cfg
        self.use_env_dynamics = cfg.use_env_dynamics
        if not self.use_env_dynamics:  # use learned model
            self.model = DynamicsModel(cfg)
            self.model.load_model(cfg.dynamics_model_ckpt)
        self.plot_rollouts = cfg.debug_cem
        if self.plot_rollouts:
            self.debug_cem_dir = os.path.join(cfg.log_dir, "debug_cem")
            os.makedirs(self.debug_cem_dir, exist_ok=True)

    def compare_optimal_actions(self, demo, start, goal, demo_name):
        """
        Run environment / model rollout on the action trajectory and compare
        the difference
        """
        old_state = self.env.get_flattened_state()
        actions = torch.from_numpy(demo["actions"]).type(torch.float32).unsqueeze(0)
        demo_start_state = demo["states"][0]
        # import ipdb; ipdb.set_trace()
        self.env.set_flattened_state(demo_start_state)
        env_rollout = generate_env_rollouts(
            self.cfg,
            self.env,
            actions,
            goal,
            opt_traj=np.zeros_like(goal.imgs),
            ret_obs=True,
        )
        model_rollout = generate_model_rollouts(
            self.cfg,
            self.env,
            self.model,
            actions,
            start,
            goal,
            opt_traj=np.zeros_like(goal.imgs),
            ret_obs=True,
        )

        env_video = env_rollout["obs"][0]
        env_cost = env_rollout["sum_cost"][0]
        model_video = np.uint8(model_rollout["obs"][0] * 255).transpose((0, 2, 3, 1))
        model_cost = model_rollout["sum_cost"][0]
        curr_img = start.img.copy()
        putText(curr_img, "START", (0, 8))
        img = np.concatenate([curr_img, curr_img, goal.imgs[0]], axis=1)
        gif = [img] * 2
        for t, (env_ob, model_ob) in enumerate(zip(env_video, model_video)):
            goal_idx = t if t < len(goal.imgs) else -1
            goal_img = goal.imgs[goal_idx]

            img = np.concatenate([env_ob, model_ob, goal_img], axis=1)
            putText(img, "ENV", (0, 8))
            putText(img, f"{env_cost:.0f}", (0, 72))
            putText(img, "SVG", (64, 8))
            putText(img, f"{model_cost:.0f}", (64, 72))
            putText(img, f"{t+1}/{len(env_video)}", (0, 124))
            putText(img, f"{t+1}/{len(env_video)}", (64, 124))
            putText(img, "GOAL", (256, 8))
            putText(img, f"{t+1}/{len(env_video)}", (128, 124))
            gif.append(img)
        gif_path = os.path.join(self.debug_cem_dir, f"{demo_name}.gif")
        imageio.mimwrite(gif_path, gif, fps=2)
        self.env.set_flattened_state(old_state)

    def get_action(self, start, goal, ep_num, step, opt_traj=None):
        """
        start: data class containing start img, start robot, etc.
        goal: data class containing goal imgs, goal robot, etc.
        ep_num: used for plotting rollouts
        step: used for plotting rollouts
        opt_traj: used for oracle demo cost
        Returns: a list of actions to execute in the environment
        """
        self.ep_num = ep_num
        self.step = step
        # Initialize action sequence belief as standard normal, of shape (L, A)
        mean = torch.zeros(self.L, self.A)
        std = torch.ones(self.L, self.A) * self.cem_init_std
        mean_top_costs = []  # for debugging
        # Optimization loop
        for _ in range(self.I):  # Use tqdm to track progress
            # Sample J candidate action sequence
            m = Normal(mean, std)
            act_seq = m.sample((self.J,))  # of shape (J, L, A)
            act_seq[-1] = 0  # always have a "do nothing" action sequence
            act_seq.clamp_(-1, 1)  # always between -1 and 1
            # Generate J rollouts
            rollouts = self._get_rollouts(act_seq, start, goal, opt_traj)
            # Select top K action sequences based on cumulative cost
            costs = torch.from_numpy(rollouts["sum_cost"])
            top_costs, top_idx = costs.topk(self.K)
            top_act_seq = torch.index_select(act_seq, dim=0, index=top_idx)
            mean_top_costs.append(f"{top_costs.mean():.3f}")

            # Update parameters for normal distribution
            std, mean = torch.std_mean(top_act_seq, dim=0)

        # Print means of top costs, for debugging
        print(
            f"\tMeans of top costs: {mean_top_costs} Opt return: {rollouts['optimal_sum_cost']:.3f}"
        )
        return mean.numpy()

    def _get_rollouts(
        self,
        act_seq,
        start: State,
        goal: DemoGoalState,
        opt_traj,
    ):
        """
        Return the rollouts either from simulator or learned model
        """
        if self.use_env_dynamics:
            rollouts = generate_env_rollouts(
                self.cfg, self.env, act_seq, goal, opt_traj=opt_traj
            )
        else:
            rollouts = generate_model_rollouts(
                self.cfg,
                self.env,
                self.model,
                act_seq,
                start,
                goal,
                ret_obs=self.plot_rollouts,
                opt_traj=opt_traj,
            )
            # Plot the Top K model rollouts
            if self.plot_rollouts:
                obs = rollouts["obs"]  # N x T x C x H x W
                obs = np.uint8(255 * obs)
                obs = obs.transpose((0, 1, 3, 4, 2))  # N x T x H x W x C
                curr_img = start.curr_img.copy()
                gif_folder = os.path.join(self.debug_cem_dir, f"ep_{self.ep_num}")
                os.makedirs(gif_folder, exist_ok=True)
                for n in range(obs.shape[0]):
                    goal_img = goal.imgs[0]
                    img = np.concatenate([curr_img, goal_img], axis=1)
                    gif = [img]
                    for t in range(self.cfg.horizon):
                        curr_img = obs[n, t]
                        g = t if t < len(goal.imgs) else -1
                        goal_img = goal.imgs[g]
                        img = np.concatenate([curr_img, goal_img], axis=1)
                        putText(img, "MODEL", (0, 8))
                        putText(img, "GOAL", (64, 8))
                        gif.append(img)
                    gif_path = os.path.join(gif_folder, f"step_{self.step}_top_{n}.gif")
                    imageio.mimwrite(gif_path, gif)
        return rollouts
