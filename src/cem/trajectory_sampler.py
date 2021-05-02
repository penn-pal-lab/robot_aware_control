from src.dataset.robonet.robonet_dataset import normalize
import time as timer
from collections import defaultdict

import numpy as np
import torch
from src.dataset.locobot.locobot_model import LocobotAnalyticalModel
from src.prediction.losses import RobotWorldCost
from src.prediction.models.dynamics import SVGConvModel
from src.utils.state import DemoGoalState, State
from src.utils.image import zero_robot_region


class TrajectorySampler(object):
    def __init__(self, cfg, model, cam_ext=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.model: SVGConvModel = model
        self.cost = RobotWorldCost(cfg)
        # use locobot analytical model to generate masks and states
        self.low = torch.from_numpy(np.array([0.015, -0.3, 0.1, 0, 0], dtype=np.float32))
        self.high = torch.from_numpy(np.array([0.55, 0.3, 0.4, 1, 1], dtype=np.float32))
        self.low.unsqueeze_(0)
        self.high.unsqueeze_(0)
        if cfg.model_use_robot_state or cfg.model_use_mask or cfg.black_robot_input:
            self.robot_model = LocobotAnalyticalModel(cfg, cam_ext=cam_ext)

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
        if not suppress_print:
            start_time = timer.time()
            print("####### Gathering Samples #######")

        # use locobot analytical model to generate masks and states
        if cfg.model_use_robot_state or cfg.model_use_mask or cfg.black_robot_input:
            '''
            states should be normalized, world frame
            '''
            states = torch.zeros((T+1, N, 5), dtype=torch.float32) # only need first timestep
            qpos = torch.zeros((T+1, N, 5), dtype=torch.float32) # only need first timestep
            states[0, :] = normalize(torch.tensor(start.state), self.low, self.high)
            qpos[0, :] = torch.tensor(start.qpos)
            start_data = {
                "states": states,
                "qpos": qpos,
                "actions": action_sequences.permute(1,0,2), # (T, N, 2)
                "low": self.low.repeat(N,1), # (N, 5)
                "high": self.high.repeat(N,1), # (N, 5)
            }
            """
            Either use thick mask for prediction, thick mask for planning or:
            normal mask for prediction, thick mask for planning
            """
            # if cfg.cem_prediction_use_thick_mask:
            #     states, masks_thick = self.robot_model.predict_batch(start_data, thick=True)
            #     masks = masks_thick = masks_thick.to(dev, non_blocking=True)
            # else:
            #     states, masks = self.robot_model.predict_batch(start_data)
            #     masks_thick = self.robot_model.predict_batch(start_data, thick=True)[1]
            #     masks = masks.to(dev, non_blocking=True)
            #     masks_thick = masks_thick.to(dev, non_blocking=True)

            states, masks = self.robot_model.predict_batch(start_data)
            states = states.to(dev, non_blocking=True)
            masks = masks_thick = masks.to(dev, non_blocking=True)


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
                if "dontcare" in cfg.reconstruction_loss or cfg.black_robot_input:
                    goal_state = State(img=goal_img, mask=goal.masks[0])
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
