import time as timer
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp
from src.prediction.models.dynamics import DynamicsModel
from torchvision.transforms.functional import to_tensor


@torch.no_grad()
def generate_model_rollouts(
    cfg,
    env,
    model: DynamicsModel,
    action_sequences,
    start_img,
    start_robot,
    start_sim,
    goal_imgs,
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
    dev = cfg.device
    N = len(action_sequences)  # Number of candidate action sequences
    T = len(action_sequences[0])
    sum_cost = np.zeros(N)
    all_obs = []  # N x T x obs
    all_step_cost = []  # N x T x 1
    goal_imgs = torch.stack([to_tensor(g) for g in goal_imgs]).to(dev)
    if cfg.demo_cost:  # for debug comparison
        optimal_traj = torch.stack([to_tensor(g) for g in cfg.optimal_traj]).to(dev)
    optimal_sum_cost = 0

    model.reset(batch_size=N)
    action_sequences = action_sequences.to(dev)

    if not suppress_print:
        start_time = timer.time()
        print("####### Gathering Samples #######")
    curr_img = to_tensor(start_img.copy()).expand(N, -1, -1, -1).to(dev)  # (N x |I|)
    curr_robot = torch.from_numpy(start_robot.copy()).expand(N, -1).to(dev)  # N x |A|)
    curr_sim = [start_sim] * N  # (N x D)
    for t in range(T):
        ac = action_sequences[:, t]  # (J, |A|)
        # compute the next img
        next_img = model.next_img(curr_img, curr_robot, ac, t == 0)
        # compute the future robot and sim using kinematics solver like mujoco
        for j in range(N):
            next_robot, next_sim = env.robot_kinematics(
                curr_sim[j], ac[j].cpu().numpy()
            )
            curr_robot[j] = torch.from_numpy(next_robot).to(dev)
            curr_sim[j] = next_sim
        # compute the img costs
        goal_idx = t if t < len(goal_imgs) else -1
        goal_img = goal_imgs[goal_idx]
        if cfg.demo_cost:  # for debug comparison
            opt_img = optimal_traj[goal_idx]

        if cfg.reward_type == "inpaint":
            rew = (
                -torch.sum((next_img - goal_img) ** 2, (1, 2, 3)).cpu().numpy()
            )  # N x 1
            if cfg.demo_cost:
                optimal_sum_cost += -torch.sum((opt_img - goal_img) ** 2).cpu().numpy()

        sum_cost += rew
        if ret_obs:
            all_obs.append(next_img)  # T x N x Obs
        if ret_step_cost:
            all_step_cost.append(rew)  # T x N x 1

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
    if ret_obs:
        rollouts["obs"] = torch.stack(all_obs).transpose(0, 1).cpu().numpy()
    if ret_step_cost:
        rollouts["step_cost"] = torch.stack(all_step_cost).transpose(0, 1).cpu().numpy()
    return rollouts


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
    sum_cost = np.zeros(N)
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
    """
    Multiprocess version of env rollouts
    In my experience, multiprocess is not faster than the serial one due to
    overhead of initializing the env in each process. Maybe choosing the right amount
    of workers and action candidate size will make parallel rollouts faster
    """
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
    """
    Create a Pool for generating rollouts
    """
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
