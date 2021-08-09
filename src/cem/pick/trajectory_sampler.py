import time as timer
from collections import defaultdict

import numpy as np
import torch
from src.prediction.losses import RobotWorldCost
from src.prediction.models.dynamics import SVGConvModel
from src.utils.state import DemoGoalState, State
from src.utils.image import zero_robot_region
from src.env.robotics.locobot_pick_env import LocobotPickEnv


class TrajectorySampler(object):
    def __init__(self, cfg, physics):
        super().__init__()
        self.cfg = cfg
        self.physics = physics
        if physics == "gt":
            self.env = LocobotPickEnv(cfg)
            self.env.reset()
        else:
            self.model: SVGConvModel = None
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
            return self.generate_env_rollouts(action_sequences, start, goal, opt_traj, ret_obs, ret_step_cost, suppress_print)
        else:
            return self.generate_model_rollouts(action_sequences, start, goal, opt_traj, ret_obs, ret_step_cost, suppress_print)

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
            # add optimal trajectory to end of action sequence list
            action_sequences = torch.cat([action_sequences, opt_traj])

        N = len(action_sequences)  # Number of candidate action sequences
        sum_cost = np.zeros(N)
        all_obs = []  # N x T x obs
        all_step_cost = []  # N x T x 1
        env_state = start.state


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
                curr_state = State(img=img, mask=mask)
                rew = 0

                # calculate cost between demonstration and current image
                goal_idx = t if t < len(goal.imgs) else -1
                goal_img = goal.imgs[goal_idx].astype(np.float32)
                goal_mask = None
                if goal.masks is not None:
                    goal_mask = goal.masks[goal_idx]
                goal_state = State(img=goal_img, mask=goal_mask)
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
        dev = cfg.device
        if opt_traj is not None:
            # (N, T, 1), (T,1)
            # pad 0s, add batch dimension to opt traj
            opt_traj = torch.cat([opt_traj, torch.zeros((len(opt_traj),3))], 1)
            opt_traj.unsqueeze_(0)
            # add optimal trajectory to end of action sequence list
            action_sequences = torch.cat([action_sequences, opt_traj])

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
        goal_masks = torch.stack([torch.from_numpy(g) for g in goal.masks]).to(dev)
        if not suppress_print:
            start_time = timer.time()
            print("####### Gathering Samples #######")

        # use locobot analytical model to generate masks and states
        if cfg.model_use_robot_state or cfg.model_use_mask or cfg.black_robot_input or "dontcare" in cfg.reward_type:
            '''
            states should be normalized, world frame
            '''
            states = torch.zeros((T+1, N, 5), dtype=torch.float32) # only need first timestep
            qpos = torch.zeros((T+1, N, cfg.robot_joint_dim), dtype=torch.float32) # only need first timestep
            start_state = torch.tensor(start.state)
            if cfg.experiment == "control_franka": # convert from franka to loco frame
                start_state[:2] = start_state[:2] + LOCO_FRANKA_DIFF

            states[0, :] = normalize(start_state, self.low, self.high)
            qpos[0, :] = torch.tensor(start.qpos)
            start_data = {
                "states": states,
                "qpos": qpos,
                "actions": action_sequences.permute(1,0,2), # (T, N, 2)
                "low": self.low.repeat(N,1), # (N, 5)
                "high": self.high.repeat(N,1), # (N, 5)
            }
            states, masks = self.robot_model.predict_batch(start_data, thick=True)
            states = states.to(dev, non_blocking=True)
            masks = masks_thick = masks.to(dev, non_blocking=True)
            # else: # calculate the future states
            #     # action is ( T, N, 2)
            #     # states is (T, N, 2)
            #     # start_state is (2)
            #     states[0, :] = start_state
            #     import ipdb; ipdb.set_trace()
            #     actions = action_sequences.permute(1,0,2)
            #     for t in range(actions.shape[0]):
            #         act = actions[t,:, :3]
            #         states[t+1, :, :3] = states[t,:, :3] + act
            #     states = normalize(states, self.low, self.high)


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
                mask, state, heatmap  = None, None, None
                if cfg.model_use_mask:
                    mask = masks[t, s:e]
                if cfg.model_use_robot_state:
                    state = states[t, s:e]
                # zero out robot pixels in input for norobot cost
                if "dontcare" in cfg.reconstruction_loss or cfg.black_robot_input:
                    curr_img = zero_robot_region(mask, curr_img)

                if cfg.model_use_future_mask:
                    mask = torch.cat([mask, masks[t+1, s:e]], 1)
                if cfg.model_use_future_robot_state:
                    state = (state, states[t+1, s:e])
                x_pred = model.forward(curr_img, mask, state, heatmap, ac, sample_mean=cfg.sample_mean)[0]
                x_pred, x_pred_mask = x_pred[:, :3], x_pred[:, 3].unsqueeze(1)
                next_img = (1 - x_pred_mask) * curr_img + (x_pred_mask) * x_pred
                if "dontcare" in cfg.reconstruction_loss or cfg.black_robot_input:
                    next_img = zero_robot_region(masks[t+1, s:e], next_img)
                # compute the img costs
                goal_idx = t if t < len(goal_imgs) else -1
                goal_img = goal_imgs[goal_idx]
                goal_mask = None
                if goal.masks is not None:
                    goal_mask = goal_masks[goal_idx]
                if "dontcare" in cfg.reconstruction_loss or cfg.black_robot_input or "dontcare" in cfg.reward_type:
                    goal_state = State(img=goal_img, mask=goal_mask)
                    curr_state = State(img=next_img, mask=masks_thick[t+1, s:e])
                else:
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
            rollouts["step_cost"] = torch.stack(all_step_cost).transpose(0, 1).cpu().numpy()
        return rollouts