import time as timer
from collections import defaultdict

import numpy as np
import torch
from src.prediction.losses import RobotWorldCost
from src.prediction.models.dynamics import SVGConvModel
from src.utils.state import DemoGoalState, State
from src.utils.image import zero_robot_region
from src.env.robotics.locobot_pick_env_mv import LocobotPickEnv


class TrajectorySampler(object):
    def __init__(self, cfg, physics):
        super().__init__()
        self.cfg = cfg
        self.physics = physics
        if physics == "gt":
            self.env = LocobotPickEnv(cfg)
            self.env.reset()
        else:
            self.model: SVGConvModel = SVGConvModel(cfg)
            ckpt = torch.load(cfg.dynamics_model_ckpt, map_location=cfg.device)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            self.env = LocobotPickEnv(cfg)
            self.env.reset()
        self.cost = RobotWorldCost(cfg)

    def generate_rollouts(
        self,
        action_sequences,
        start: State,
        goal: DemoGoalState,
        opt_traj=None,
        ret_obs=False,
        ret_step_cost=False,
        suppress_print=True,
    ):
        if self.physics == "gt":
            return self.generate_env_rollouts(
                action_sequences,
                start,
                goal,
                opt_traj,
                ret_obs,
                ret_step_cost,
                suppress_print,
            )
        else:
            return self.generate_model_rollouts(
                action_sequences,
                start,
                goal,
                opt_traj,
                ret_obs,
                ret_step_cost,
                suppress_print,
            )

    def generate_env_rollouts(
        self,
        action_sequences,
        start: State,
        goal: DemoGoalState,
        opt_traj=None,
        ret_obs=False,
        ret_step_cost=False,
        suppress_print=True,
    ):
        cfg = self.cfg
        env = self.env
        cost = RobotWorldCost(self.cfg)
        T = len(action_sequences[0])
        if opt_traj is not None:
            # (N, T, 1), (T,1)
            opt_traj = torch.from_numpy(opt_traj[:T][None])
            # add no-op actions if horizon is greater than opt action sequence
            opt_traj = torch.cat(
                [opt_traj, torch.zeros((1, T - opt_traj.shape[1], 4))], 1
            )
            # add optimal trajectory to end of action sequence list
            action_sequences = torch.cat([action_sequences, opt_traj])

        N = len(action_sequences)  # Number of candidate action sequences
        sum_cost = np.zeros(N)
        all_obs = []  # N x T x obs
        all_step_cost = []  # N x T x 1
        env_state = start.sim_state

        if not suppress_print:
            start_time = timer.time()
            print("####### Gathering Samples #######")
        # rollout action sequence per episode
        for ep_num in range(N):
            ep_obs = []
            ep_cost = []
            env.set_flattened_state(env_state.copy())

            for t in range(T):
                action = action_sequences[ep_num, t].numpy()
                ob, _, _, _ = env.step(action)

                img = ob["observation"].astype(np.float32)
                mask = ob["masks"]
                state = ob["states"]
                if cfg.reward_type == "dontcare":
                    img = zero_robot_region(mask, img)
                curr_state = State(img=img, mask=mask, state=state)
                rew = 0

                # calculate cost between demonstration and current image
                goal_idx = t if t < len(goal.imgs) else -1
                goal_img = goal.imgs[goal_idx].astype(np.float32)
                goal_mask = None
                if goal.masks is not None:
                    goal_mask = goal.masks[goal_idx]
                    if cfg.reward_type == "dontcare":
                        goal_img = zero_robot_region(goal_mask, goal_img)

                if goal.states is not None:
                    goal_state = goal.states[goal_idx]

                goal_state = State(img=goal_img, mask=goal_mask, state=goal_state)
                if not cfg.sparse_cost or (cfg.sparse_cost and t == T - 1):
                    rew = cost(curr_state, goal_state, print_cost=False)

                sum_cost[ep_num] += rew
                if ret_obs:
                    ep_obs.append(img)
                if ret_step_cost:
                    ep_cost.append(rew)

            all_obs.append(ep_obs)
            all_step_cost.append(ep_cost)

        if not suppress_print:
            print(
                "======= Samples Gathered  ======= | >>>> Time taken = %f "
                % (timer.time() - start_time)
            )

        # debug: print like 25 gifs
        # import imageio
        # for i, episode in enumerate(all_obs[:10]):
        #     imageio.mimwrite(f"cem_{i}.gif", episode)
        # import ipdb; ipdb.set_trace()

        rollouts = defaultdict(float)
        if opt_traj is not None:
            rollouts["optimal_sum_cost"] = sum_cost[-1]
            rollouts["optimal_obs"] = np.asarray(all_obs[-1])
            # ignore the optimal trajectory cost for argsort
            sum_cost = sum_cost[:-1]

        rollouts["sum_cost"] = sum_cost
        # just return the top K trajectories
        if ret_obs:
            topk_idx = np.argsort(sum_cost)[: cfg.topk]
            all_obs = np.asarray(all_obs)
            topk_obs = all_obs[topk_idx]
            rollouts["topk_idx"] = topk_idx
            rollouts["obs"] = topk_obs
        if ret_step_cost:
            rollouts["step_cost"] = np.asarray(all_step_cost)

        return rollouts

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

        action_sequences: list of action candidates to evaluate
        cost: cost function for comparing observation and goal
        goal: list of goal images for comparison
        ret_obs: return the observations
        ret_step_cost: return per step cost
        start: not normalized states

        Returns a dictionary containing the cost per trajectory by default.
        """
        cfg = self.cfg
        model = self.model
        cost = self.cost
        env = self.env
        dev = cfg.device
        T = len(action_sequences[0])
        # print(action_sequences.dtype)
        if opt_traj is not None:
            # (N, T, 1), (T,1)
            opt_traj = torch.from_numpy(opt_traj[:T][None])
            # add no-op actions if horizon is greater than opt action sequence
            opt_traj = torch.cat(
                [opt_traj, torch.zeros((1, T - opt_traj.shape[1], 4))], 1
            )
            # add optimal trajectory to end of action sequence list
            action_sequences = torch.cat([action_sequences, opt_traj])

        N = len(action_sequences)  # Number of candidate action sequences
        action_sequences = torch.cat([action_sequences, torch.zeros((N,T,1))], -1)
        action_sequences[:, :, 4] = action_sequences[:,:,3]
        action_sequences[:, :, 3] = 0
        action_sequences = action_sequences.float()
        # print(action_sequences.dtype)
        # import ipdb; ipdb.set_trace()

        ac_per_batch = cfg.candidates_batch_size
        B = max(N // ac_per_batch, 1)  # number of candidate batches per GPU pass
        sum_cost = np.zeros(N)
        all_obs = torch.zeros((N, T, 3, 48, 64 * 2))  # N x T x obs
        all_step_cost = np.zeros((N, T))  # N x T x 1
        goal_imgs = torch.stack(
            [torch.from_numpy(g).permute(2, 0, 1).float() / 255 for g in goal.imgs]
        ).to(dev)
        goal_masks = torch.stack([torch.from_numpy(g) for g in goal.masks]).to(dev)
        if not suppress_print:
            start_time = timer.time()
            print("####### Gathering Samples #######")

        for b in range(B):
            s = b * ac_per_batch
            e = (b + 1) * ac_per_batch if b < B - 1 else N
            num_batch = e - s
            action_batch = action_sequences[s:e]
            actions = action_batch.to(dev)
            model.init_hidden(batch_size=num_batch)
            curr_img = torch.from_numpy(start.img.copy()).permute(2, 0, 1).float() / 255
            curr_img = curr_img.expand(num_batch, -1, -1, -1).to(dev)  # (N x |I|)
            sim_states = [start.sim_state.copy() for _ in range(num_batch)]
            mask = torch.from_numpy(start.mask).unsqueeze(0).repeat(num_batch, 1, 1 ,1).float().to(dev)
            state = torch.from_numpy(start.state).repeat(num_batch, 1).float().to(dev)
            for t in range(T):
                ac = actions[:, t]  # (J, |A|)
                # zero out robot pixels in input for norobot cost
                if "dontcare" in cfg.reconstruction_loss or cfg.black_robot_input:
                    curr_img = zero_robot_region(mask, curr_img)
                # compute the next img
                x_pred = model.forward(
                    curr_img, mask, state, None, ac, sample_mean=cfg.sample_mean
                )[0]
                x_pred, x_pred_mask = x_pred[:, :3], x_pred[:, 3].unsqueeze(1)
                next_img = (1 - x_pred_mask) * curr_img + (x_pred_mask) * x_pred
                # compute the state and mask for the next iteration using mujoco.
                if cfg.model_use_mask or cfg.model_use_robot_state:
                    for n in range(num_batch):
                        env.set_flattened_state(sim_states[n])
                        a = action_batch[n, t, (0,1,2,4)].numpy()
                        next_ob, *_ = env.step(a)
                        mask[n] = torch.from_numpy(next_ob["masks"])
                        state[n] = torch.from_numpy(next_ob["states"])
                        sim_states[n] = env.get_flattened_state().copy()
                    mask = mask.float().to(dev)
                    state = state.float().to(dev)

                if "dontcare" in cfg.reconstruction_loss or cfg.black_robot_input:
                    next_img = zero_robot_region(mask, next_img)
                curr_state = State(img=next_img, mask=mask, state=state.cpu())

                # compute the img costs
                rew = 0
                goal_idx = t if t < len(goal_imgs) else -1
                goal_img = goal_imgs[goal_idx]
                goal_mask = None
                if goal.masks is not None:
                    goal_mask = goal_masks[goal_idx] # (H, W)
                    goal_mask = goal_mask.unsqueeze(0).unsqueeze(0).repeat(num_batch,1,1,1) # (N, 1, H, W)
                    goal_img = goal_img.unsqueeze(0).repeat(num_batch, 1,1,1) # (N, 3, H, W)
                    if cfg.reward_type == "dontcare" or "dontcare" in cfg.reconstruction_loss or cfg.black_robot_input:
                        goal_img = zero_robot_region(goal_mask, goal_img)
                if goal.states is not None:
                    goal_state = goal.states[goal_idx]
                goal_state = State(img=goal_img, mask=goal_mask, state=goal_state)
                # sparse_cost only uses last frame of trajectory for cost
                if not cfg.sparse_cost or (cfg.sparse_cost and t == T - 1):
                    rew = cost(curr_state, goal_state, print_cost=False)
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
        if opt_traj is not None:
            rollouts["optimal_sum_cost"] = sum_cost[-1]
            rollouts["optimal_obs"] = all_obs[-1].cpu().numpy()
            # ignore the optimal trajectory cost
            sum_cost = sum_cost[:-1]

        rollouts["sum_cost"] = sum_cost
        # just return the top K trajectories
        if ret_obs:
            topk_idx = np.argsort(sum_cost)[: cfg.topk]
            topk_obs = all_obs[topk_idx]
            rollouts["topk_idx"] = topk_idx
            rollouts["obs"] = topk_obs.cpu().numpy()
        if ret_step_cost:
            rollouts["step_cost"] = (
                torch.stack(all_step_cost).transpose(0, 1).cpu().numpy()
            )
        return rollouts