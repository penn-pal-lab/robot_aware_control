import torch
from src.cem.trajectory_sampler import generate_env_rollouts, generate_model_rollouts
from src.prediction.models.dynamics import DynamicsModel
from torch.distributions.normal import Normal


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
        curr_img: used by learned model, not needed for ground truth model
        curr_robot: robot eef pos, used by learned model, not needed for ground truth model
        curr_sim: used by learned model, not needed for ground truth model
        Goal_imgs: set of goals for computing costs
        Returns: a list of actions to execute in the environmetn
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
            rollouts = self._get_rollouts(
                act_seq, curr_img, curr_robot, curr_sim, goal_imgs
            )
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

    def _get_rollouts(self, act_seq, curr_img, curr_robot, curr_sim, goal_imgs):
        """
        Return the rollouts either from simulator or learned model
        """
        if self.use_env_dynamics:
            rollouts = generate_env_rollouts(self.cfg, self.env, act_seq, goal_imgs)
        else:
            rollouts = generate_model_rollouts(
                self.cfg,
                self.env,
                self.model,
                act_seq,
                curr_img,
                curr_robot,
                curr_sim,
                goal_imgs,
            )
        return rollouts