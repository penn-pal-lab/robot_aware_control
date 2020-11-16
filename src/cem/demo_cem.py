import numpy as np
import torch
from src.prediction.losses import InpaintBlurCost
from src.cem.trajectory_sampler import generate_env_rollouts, generate_model_rollouts
from torch.distributions.normal import Normal
from torchvision.transforms import ToTensor
from src.prediction.models.dynamics import DynamicsModel


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

        # Infer action size
        self.A = env.action_space.shape[0]
        self.env = env
        self.cfg = cfg
        self.use_env_dynamics = cfg.use_env_dynamics
        if not self.use_env_dynamics:  # use learned model
            self.model = DynamicsModel(cfg)
            self.model.load_model(cfg.dynamics_model_ckpt)

    def get_action(self, curr_img, curr_robot, curr_sim, goal_imgs):
        """
        Use actual physics engine as dynamics model for CEM.
        curr_img: used by learned model, not needed for ground truth model
        curr_robot: robot eef pos, used by learned model, not needed for ground truth model
        curr_sim: used by learned model, not needed for ground truth model
        Goal_imgs: set of goals for computing costs
        """
        # Initialize action sequence belief as standard normal, of shape (L, A)
        mean = torch.zeros(self.L, self.A)
        std = torch.ones(self.L, self.A)
        mean_top_costs = []  # for debugging
        # Optimization loop
        for _ in range(self.I):  # Use tqdm to track progress
            # Sample J candidate action sequence
            m = Normal(mean, std)
            act_seq = m.sample((self.J,))  # of shape (J, L, A)
            # Generate J rollouts
            if self.use_env_dynamics:
                rollouts = generate_env_rollouts(self.cfg, self.env, act_seq, goal_imgs)
            else:
            # TODO: generate learned dynamics CEM here
                rollouts = generate_model_rollouts(self.cfg, self.env, act_seq, goal_imgs)

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
        # Return first R actions, where R is number of actions to take before replanning
        return mean[: self.R, :].numpy()


# def cem_env_policy(env, goal_imgs, cfg):
#     """
#     Use actual physics engine as dynamics model for CEM
#     """
#     # Hyperparameters
#     L = cfg.horizon  # Prediction window size
#     I = cfg.opt_iter  # Number of optimization iterations
#     J = cfg.action_candidates  # Number of candidate action sequences
#     K = cfg.topk

#     # Infer action size
#     A = env.action_space.shape[0]

#     # Initialize action sequence belief as standard normal, of shape (L, A)
#     mean = torch.zeros(L, A)
#     std = torch.ones(L, A)
#     mean_top_costs = []  # for debugging
#     # Optimization loop
#     for i in range(I):  # Use tqdm to track progress
#         # Sample J candidate action sequence
#         m = Normal(mean, std)
#         act_seq = m.sample((J,))  # of shape (J, L, A)
#         # Generate J rollouts
#         rollouts = generate_env_rollouts(cfg, env, act_seq, goal_imgs)

#         # Select top K action sequences based on cumulative cost
#         costs = torch.from_numpy(rollouts["sum_cost"])
#         top_costs, top_idx = costs.topk(K)
#         top_act_seq = torch.index_select(act_seq, dim=0, index=top_idx)
#         mean_top_costs.append(f"{top_costs.mean():.3f}")

#         # Update parameters for normal distribution
#         std, mean = torch.std_mean(top_act_seq, dim=0)

#     # Print means of top costs, for debugging
#     print(
#         f"\tMeans of top costs: {mean_top_costs} Opt return: {rollouts['optimal_sum_cost']:.3f}"
#     )
#     # save gifs of top trajectories for debugging
#     # Return first action mean, of shape (A)
#     return mean[: cfg.replan_every, :]