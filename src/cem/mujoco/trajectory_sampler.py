from src.prediction.losses import RobotWorldCost
import time as timer
from collections import defaultdict
from src.utils.state import State, DemoGoalState

import numpy as np
import torch
from torch import cat
import torch.multiprocessing as mp
from src.prediction.models.dynamics import DynamicsModel


@torch.no_grad()
def generate_model_rollouts(
    cfg,
    env,
    model: DynamicsModel,
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
    all_obs = torch.zeros((N, T, 3, 128, 64))  # N x T x obs
    all_step_cost = np.zeros((N, T))  # N x T x 1
    goal_imgs = torch.stack(
        [torch.from_numpy(g).permute(2, 0, 1).float() / 255 for g in goal.imgs]
    ).to(dev)
    goal_masks = torch.stack([torch.from_numpy(g) for g in goal.masks]).to(dev)
    start_mask = torch.from_numpy(start_state.mask).unsqueeze_(0)
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
        model.reset(batch_size=num_batch)
        curr_img = cat(
            [
                torch.from_numpy(start_state.img.copy()).permute(2, 0, 1).float() / 255,
                start_mask,
            ],
            dim=0,
        )
        curr_img = curr_img.expand(num_batch, -1, -1, -1).to(dev)  # (N x |I|)
        curr_robot = (
            torch.from_numpy(start_state.robot.copy()).expand(num_batch, -1).to(dev)
        )  # N x |A|)
        curr_sim = [start_state.sim] * num_batch  # (N x D)
        curr_mask = torch.zeros((num_batch, 1, 128, 64), dtype=torch.bool).to(
            dev
        )  # (N x 1 x H x W)
        for t in range(T):
            ac = actions[:, t]  # (J, |A|)
            # compute the next img
            use_skip = t % 1 == 0
            next_img = model.next_img(curr_img, curr_robot, ac, use_skip)
            # compute the future robot dynamics using mujoco
            for j in range(num_batch):
                next_robot, next_mask, next_sim = env.robot_kinematics(
                    curr_sim[j], ac[j].cpu().numpy(), ret_mask=True
                )
                curr_robot[j] = torch.from_numpy(next_robot).to(dev)
                curr_sim[j] = next_sim
                curr_mask[j] = torch.from_numpy(next_mask).unsqueeze_(0).to(dev)
            # compute the img costs
            goal_idx = t if t < len(goal_imgs) else -1
            goal_img = goal_imgs[goal_idx]
            goal_robot = goal.robots[goal_idx]
            goal_mask = None
            if goal.masks is not None:
                goal_mask = goal_masks[goal_idx]
            goal_state = State(img=goal_img, robot=goal_robot, mask=goal_mask)
            curr_state = State(img=next_img, robot=curr_robot, mask=curr_mask)
            rew = 0
            # sparse_cost only uses last frame of trajectory for cost
            if not cfg.sparse_cost or (cfg.sparse_cost and t == T - 1):
                rew = cost(curr_state, goal_state)
            sum_cost[start:end] += rew
            if ret_obs:
                all_obs[start:end, t] = next_img.cpu()  # B x T x Obs
            if ret_step_cost:
                all_step_cost[start:end] = rew  # B x T x 1
            curr_img = cat([next_img, curr_mask], dim=1)

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


def generate_env_rollouts(
    cfg,
    env,
    action_sequences,
    goal: DemoGoalState,
    ret_obs=False,
    ret_step_cost=False,
    suppress_print=True,
    opt_traj=None,
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
    cost = RobotWorldCost(cfg)
    N = len(action_sequences)  # Number of candidate action sequences
    T = len(action_sequences[0])
    sum_cost = np.zeros(N)
    all_obs = []  # N x T x obs
    all_step_cost = []  # N x T x 1
    if env._background_img is not None:
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
            goal_idx = t if t < len(goal.imgs) else -1
            goal_img = goal.imgs[goal_idx].astype(np.float32)
            goal_robot = goal.robots[goal_idx]
            goal_mask = None
            if goal.masks is not None:
                goal_mask = goal.masks[goal_idx]
            goal_state = State(img=goal_img, robot=goal_robot, mask=goal_mask)
            if opt_traj is not None:  # for debug comparison
                opt_state = opt_traj[goal_idx]

            action = action_sequences[ep_num, t].numpy()
            ob, _, _, _ = env.step(action)

            img = ob["observation"].astype(np.float32)
            robot = ob["robot"]
            mask = ob["mask"]
            curr_state = State(img=img, robot=robot, mask=mask)
            rew = 0
            if not cfg.sparse_cost or (cfg.sparse_cost and t == T - 1):
                rew = cost(curr_state, goal_state)
                if opt_traj is not None and ep_num == 0:
                    optimal_sum_cost += cost(opt_state, goal_state)

                    # print("env", optimal_sum_cost, goal_idx)
                    # import pickle
                    # with open(f"env_goal_{t}.pkl", "wb") as f:
                    #     data = {"opt_img": opt_img, "goal_img": goal_img}
                    #     pickle.dump(data, f)

            sum_cost[ep_num] += rew
            if ret_obs:
                ep_obs.append(img)
            if ret_step_cost:
                ep_cost.append(rew)

        all_obs.append(ep_obs)
        all_step_cost.append(ep_cost)
        env.set_flattened_state(env_state.copy())  # reset env to before rollout
        if env._background_img is not None:
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
