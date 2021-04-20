import time as timer
from collections import defaultdict

import numpy as np
import torch
from src.dataset.locobot.locobot_model import LocobotAnalyticalModel
from src.prediction.losses import RobotWorldCost
from src.prediction.models.dynamics import SVGConvModel
from src.utils.state import DemoGoalState, State


class TrajectorySampler(object):
    def __init__(self, cfg, model) -> None:
        super().__init__()
        self.cfg = cfg
        self.model: SVGConvModel = model
        self.cost = RobotWorldCost(cfg)
        # use locobot analytical model to generate masks and states
        self.low = torch.from_numpy(np.array([0.015, -0.3, 0.1, 0, 0], dtype=np.float32))
        self.high = torch.from_numpy(np.array([0.55, 0.3, 0.4, 1, 1], dtype=np.float32))
        self.low.unsqueeze_(0)
        self.high.unsqueeze_(0)
        # if cfg.model_use_robot_state or cfg.model_use_mask:
        if True:
            self.robot_model = LocobotAnalyticalModel(cfg)

    @torch.no_grad()
    def generate_model_rollouts(
        self,
        action_sequences,
        start: State,
        goal: DemoGoalState,
        opt_traj=None,
        ret_obs=False,
        ret_step_cost=False,
        suppress_print=True,
    ):
        """
        Executes the action sequences on the learned model.

        cfg: configuration dictionary
        env: environment instance
        action_sequences: list of action candidates to evaluate
        cost: cost function for comparing observation and goal
        goal: list of goal images for comparison
        ret_obs: return the observations
        ret_step_cost: return per step cost

        Returns a dictionary containing the cost per trajectory by default.
        """
        cfg = self.cfg
        model = self.model
        cost = self.cost
        dev = cfg.device
        N = len(action_sequences)  # Number of candidate action sequences
        ac_per_batch = cfg.candidates_batch_size
        B = max(N // ac_per_batch, 1)  # number of candidate batches per GPU pass
        T = len(action_sequences[0])
        sum_cost = np.zeros(N)
        all_obs = torch.zeros((N, T, 3, 48, 64))  # N x T x obs
        all_step_cost = np.zeros((N, T))  # N x T x 1
        goal_imgs = torch.stack(
            [torch.from_numpy(g).permute(2, 0, 1).float() / 255 for g in goal.imgs]
        ).to(dev)
        optimal_sum_cost = 0
        if not suppress_print:
            start_time = timer.time()
            print("####### Gathering Samples #######")

        # use locobot analytical model to generate masks and states
        #if cfg.model_use_robot_state or cfg.model_use_mask:
        if True:
            '''
            states should be normalized, world frame
            '''
            states = torch.zeros((T+1, N, 5), dtype=torch.float32) # only need first timestep
            qpos = torch.zeros((T+1, N, 5), dtype=torch.float32) # only need first timestep
            states[0, :] = start.state
            qpos[0, :] = start.qpos
            start_data = {
                "states": states,
                "qpos": qpos,
                "actions": action_sequences.permute(1,0,2), # (T, N, 2)
                "low": self.low.repeat(N,1), # (N, 5)
                "high": self.high.repeat(N,1), # (N, 5)
                "masks": torch.zeros((T+1,N,1, 48, 64)), # only shape is used in predict batch
            }
            states, masks = self.robot_model.predict_batch(start_data)

        for b in range(B):
            s = b * ac_per_batch
            e = (b + 1) * ac_per_batch if b < B - 1 else N
            num_batch = e - s
            action_batch = action_sequences[s:e]
            actions = action_batch.to(dev)
            model.init_hidden(batch_size=num_batch)
            curr_img = torch.from_numpy(start.img.copy()).permute(2, 0, 1).float() / 255
            curr_img = curr_img.expand(num_batch, -1, -1, -1).to(dev)  # (N x |I|)
            for t in range(T):
                ac = actions[:, t]  # (J, |A|)
                # compute the next img
                mask, robot, heatmap, next_image, next_mask, next_robot, next_heatmap  = None, None, None, None, None, None, None
                # TODO: use z_mean instead of z_sample
                x_pred = model.forward(curr_img, mask, robot, heatmap, ac, next_image, next_mask, next_robot, next_heatmap)[0]
                x_pred, x_pred_mask = x_pred[:, :3], x_pred[:, 3].unsqueeze(1)
                next_img = (1 - x_pred_mask) * curr_img + (x_pred_mask) * x_pred

                # compute the img costs
                goal_idx = t if t < len(goal_imgs) else -1
                goal_img = goal_imgs[goal_idx]
                goal_state = State(img=goal_img)
                curr_state = State(img=next_img)
                rew = 0
                # sparse_cost only uses last frame of trajectory for cost
                if not cfg.sparse_cost or (cfg.sparse_cost and t == T - 1):
                    rew = cost(curr_state, goal_state)
                sum_cost[s:e] += rew
                if ret_obs:
                    all_obs[s:e, t] = next_img.cpu()  # B x T x Obs
                if ret_step_cost:
                    all_step_cost[s:e] = rew  # B x T x 1
                curr_img = next_img

        if not suppress_print:
            print(
                "======= Samples Gathered  ======= | >>>> Time taken = %f "
                % (timer.time() - start_time)
            )

        rollouts = defaultdict(float)
        rollouts["sum_cost"] = sum_cost
        if cfg.demo_cost:
            rollouts["optimal_sum_cost"] = optimal_sum_cost
        # just return the top K trajectories
        if ret_obs:
            topk_idx = np.argsort(sum_cost)[: cfg.topk]
            topk_obs = all_obs[topk_idx]
            rollouts["obs"] = topk_obs.cpu().numpy()
        if ret_step_cost:
            rollouts["step_cost"] = torch.stack(all_step_cost).transpose(0, 1).cpu().numpy()
        return rollouts
