import time as timer
from collections import defaultdict

import numpy as np
import torch
from src.prediction.losses import RobotWorldCost
from src.prediction.models.dynamics import SVGConvModel
from src.utils.state import DemoGoalState, State
from torch import cat


@torch.no_grad()
def generate_model_rollouts(
    cfg,
    model: SVGConvModel,
    action_sequences,
    start_state: State,
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
    cost = RobotWorldCost(cfg)
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
    for b in range(B):
        start = b * ac_per_batch
        end = (b + 1) * ac_per_batch if b < B - 1 else N
        num_batch = end - start
        action_batch = action_sequences[start:end]
        actions = action_batch.to(dev)
        model.init_hidden(batch_size=num_batch)
        curr_img = torch.from_numpy(start_state.img.copy()).permute(2, 0, 1).float() / 255
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
            sum_cost[start:end] += rew
            if ret_obs:
                all_obs[start:end, t] = next_img.cpu()  # B x T x Obs
            if ret_step_cost:
                all_step_cost[start:end] = rew  # B x T x 1
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
