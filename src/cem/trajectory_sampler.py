from collections import defaultdict

import numpy as np
import torch


def env_rollout_runner(
    cfg, env, action_sequences, cost, goal_imgs, ret_obs=False, ret_step_cost=False
):
    """
    Executes the action sequences on the environment. Used by the ground truth
    dynamics CEM to generate trajectories for action selection.

    action_sequences: list of action candidates to evaluate
    cost: cost function for comparing observation and goal
    goal: list of goal images for comparison
    ret_obs: return the observations
    ret_step_cost: return per step cost

    Returns a dictionary containing the cost per trajectory by default.
    """
    N = len(action_sequences)  # Number of candidate action sequences
    T = len(action_sequences[0])
    sum_cost = torch.zeros(N)
    all_obs = []  # N x T x obs
    all_step_cost = []  # N x T x 1

    bg_img = env._background_img.copy()
    env_state = env.get_flattened_state()
    for ep_num in range(N):
        ep_obs = []
        ep_cost = []
        optimal_sum_cost = 0
        for t in range(T):
            goal_idx = t if t < len(goal_imgs) else -1
            goal_img = goal_imgs[goal_idx]
            if cfg.demo_cost:  # for debug comparison
                opt_img = cfg.optimal_traj[goal_idx]

            action = action_sequences[ep_num, t].numpy()
            ob, _, _, _ = env.step(action)

            img = ob["observation"]
            if cfg.reward_type == "inpaint-blur":
                blur = t < T - cfg.unblur_timestep
                rew = cost(img, goal_img, blur)
            elif cfg.reward_type == "inpaint":
                rew = -np.linalg.norm(img - goal_img)
                if cfg.demo_cost:
                    optimal_sum_cost += -np.linalg.norm(opt_img - goal_img)

            sum_cost[ep_num] += rew
            if ret_obs:
                ep_obs.append(img)
            if ret_step_cost:
                ep_cost.append(rew)

        all_obs.append(ep_obs)
        all_step_cost.append(ep_cost)
        env.set_flattened_state(env_state.copy())  # reset env to before rollout
        env._background_img = bg_img.copy()

    rollouts = defaultdict(float)
    rollouts["sum_cost"] = sum_cost
    if cfg.demo_cost:
        rollouts["optimal_sum_cost"] = optimal_sum_cost
    if ret_obs:
        rollouts["obs"] = np.asarray(all_obs)
    if ret_step_cost:
        rollouts["step_cost"] = np.asarray(all_step_cost)
    return rollouts
