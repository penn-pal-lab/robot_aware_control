from collections import defaultdict

import time as timer
import multiprocessing as mp
import numpy as np
import torch
from copy import deepcopy


def generate_env_rollouts(
    cfg,
    env,
    action_sequences,
    goal_imgs,
    ret_obs=False,
    ret_step_cost=False,
    suppress_print=True,
):
    """
    Executes the action sequences on the environment. Used by the ground truth
    dynamics CEM to generate trajectories for action selection.

    cfg: configuration dictionary
    env: environment instance
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
    if not suppress_print:
        start_time = timer.time()
        print("####### Gathering Samples #######")
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
            if cfg.reward_type == "inpaint":
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
    if not suppress_print:
        print(
            "======= Samples Gathered  ======= | >>>> Time taken = %f "
            % (timer.time() - start_time)
        )

    rollouts = defaultdict(float)
    rollouts["sum_cost"] = sum_cost
    if cfg.demo_cost:
        rollouts["optimal_sum_cost"] = optimal_sum_cost
    if ret_obs:
        rollouts["obs"] = np.asarray(all_obs)
    if ret_step_cost:
        rollouts["step_cost"] = np.asarray(all_step_cost)
    return rollouts


def generate_env_rollouts_parallel(
    cfg,
    env,
    action_sequences,
    goal_imgs,
    ret_obs=False,
    ret_step_cost=False,
    max_process_time=500,
    max_timeouts=1,
    suppress_print=True,
):

    num_cpu = mp.cpu_count()
    N = len(action_sequences)
    actions_per_cpu = N // num_cpu
    args_list = []
    for i in range(num_cpu):
        start = i * actions_per_cpu
        end = (i + 1) * actions_per_cpu if i < num_cpu - 1 else N
        cpu_action_sequences = action_sequences[start:end]
        args_list_cpu = (
            cfg,
            env,
            cpu_action_sequences,
            goal_imgs,
            ret_obs,
            ret_step_cost,
        )
        args_list.append(args_list_cpu)

    # Do multiprocessing
    if not suppress_print:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts)

    # result is paths type, and results is list of paths
    rollouts = defaultdict(float)
    for result in results:
        rollouts["sum_cost"] = result["sum_cost"]
        rollouts["optimal_sum_cost"] = result["optimal_sum_cost"]
        rollouts["obs"] = result["obs"]
        rollouts["step_cost"] = result["step_cost"]

    if not suppress_print:
        print(
            "======= Samples Gathered  ======= | >>>> Time taken = %f "
            % (timer.time() - start_time)
        )

    return rollouts


def _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts):

    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
    parallel_runs = [
        pool.apply_async(generate_env_rollouts, args=args_list[i])
        for i in range(num_cpu)
    ]

    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts - 1)

    pool.close()
    pool.terminate()
    pool.join()
    return results
