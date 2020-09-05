import os
from collections import defaultdict

import numpy as np
import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions.normal import Normal

from src.env.fetch.fetch_push import FetchPushEnv



# Use actual gym environment to test cem algorithm
def cem_planner(env, config):
    # Hyperparameters
    L = config.horizon      # Prediction window size
    I = config.opt_iter      # Number of optimization iterations
    J = config.action_candidates    # Number of candidate action sequences
    K = config.topk      # Number of top K candidate action sequences to select for optimization


    # Infer action size
    A = env.action_space.shape[0]

    # Initialize action sequence belief as standard normal, of shape (L, A)
    mean = torch.zeros(L, A)
    std = torch.ones(L, A)

    ret_topks = []  # for debugging
    # Optimization loop
    for i in range(I):     # Use tqdm to track progress
        # Sample J candidate action sequence
        m = Normal(mean, std)
        act_seq = m.sample((J,))    # of shape (J, L, A)

        # Generate J rollouts
        ret_preds = torch.zeros(J)
        for j in range(J):
            # Copy environment with its state, goal, and set to dense reward
            # use set_state and get_state
            env_state = env.get_state()
            for l in range(L):
                action = act_seq[j, l].numpy()
                _, rew, _, _ = env.step(action)    # Take one step
                ret_preds[j] += rew     # accumulate rewards
            env.set_state(env_state)  # reset env to before rollout

        # Select top K action sequences based on cumulative rewards
        ret_topk, idx = ret_preds.topk(K)
        top_act_seq = torch.index_select(act_seq, dim=0, index=idx)     # of shape (K, L, A)
        ret_topks.append("%.3f" % ret_topk.mean())     # Record mean of top returns

        # Update parameters for normal distribution
        std, mean = torch.std_mean(top_act_seq, dim=0)

    # Print means of top returns, for debugging
    # print("\tMeans of top returns: ", ret_topks)
    # Return first action mean, of shape (A)
    return mean[0, :]


def run_cem_episodes(config):
    num_episodes = config.num_episodes
    video_folder = os.path.join(os.path.dirname(__file__), f"../../../{config.prefix}_videos")
    os.makedirs(video_folder, exist_ok=True)
    env = FetchPushEnv(config)
    # Do rollouts of CEM control
    all_episode_stats = defaultdict(list)
    success_record = np.zeros(num_episodes)
    for i in range(num_episodes): # this can be parallelized
        ep_history = defaultdict(int)
        env.reset()
        vr = VideoRecorder(env, metadata=None, path=f'{video_folder}/test_{i}.mp4')

        ret = 0     # Episode return
        s = 0       # Step count
        print("\n=== Episode %d ===\n" % (i+1))
        while True:
            print("\tStep {}".format(s))
            action = cem_planner(env, config).numpy()       # Action convert to numpy array
            obs, rew, done, info = env.step(action)
            ret += rew
            s += 1
            vr.capture_frame()
            # store info history
            for k, v in info.items():
                ep_history[k] += float(v)

            # Calculate distance to goal at current step
            print("\tReward: {}".format(rew))

            if info['is_success'] > 0 or done or s > 10:
                print("=" * 10 + f"Episode {i}" + "=" * 10)
                for k, v in ep_history.items():
                    print(f"{k}: {v}")
                    all_episode_stats[k].append(v)
                break
        vr.close()

    # Close video recorder
    env.close()

    # Summary
    print("\n\n### Summary ###")
    for k, v in all_episode_stats.items():
        print(f"{k} avg: {np.mean(v)} \u00B1 {np.std(v)}")

if __name__ == "__main__":
    from src.config import argparser
    config, _ = argparser()
    run_cem_episodes(config)

# class DynamicsModel(object):
#     """
#     Base class for a future prediction model P(S' | S, A)
#     """
#     def predict(self, state, action):
#         return NotImplementedError

# class MujocoDynamicsModel(DynamicsModel):
#     """Uses mujoco sim as ground truth"""
#     def __init__(self, config, sim):
#         self.sim = sim

#     def predict(self, state, action):
#         """
#         State is a ()
#         Returns (num_cand, horizon) array of returns
#         """



# class CEMPlanner(object):
#     """
#         Carries out Cross-Entropy Method (CEM) for model-based planning.
#         Draw action sequences from an parameterized distributions, generate states rollouts using provided state
#             predictive model, evaluates action sequences using reward predictive model, and fit distrition parameters on
#             N best action sequences using CEM.
#     """

#     def __init__(self, cond_len=5, plan_len=10, num_iter=20, num_cand=128, topk=8):
#         """

#         :param cond_len:  conditional window length := T
#         :param plan_len:  planning window length := L
#         :param num_iter:  number of optimization iterations := I
#         :param num_cand:  number of candidates per iteration := J
#         :param topk:  the top K candidates := K
#         """
#         self.T = cond_len
#         self.L = plan_len
#         self.I = num_iter
#         self.J = num_cand
#         self.K = topk

#     def plan_single_step(self, pred_model, cond_frames, cond_agstate, cond_actions):
#         """

#         :param pred_model:  predictive model (state + reward prediction)
#         :param cond_frames:  conditional frmaes. Tensor of shape (1, T, C, W, H)
#         :param cond_agstate:  conditional agent states. Tensor of shape (1, T, S) where S is the size of one single
#                                 agent state
#         :param cond_actions:  conditional actions, i.e., the historical actions the agent took when interacting with the
#                                 environment within this conditional window. Tensor of shape (1, T, A) where A is the
#                                 size of actions at one timestep.
#         :return:
#         """
#         # Infer action size
#         A = cond_actions.shape[-1]

#         # Initialize action sequence belief as standard normal, of shape (L, A)
#         mean = torch.zeros(self.L, A)
#         std = torch.ones(self.L, A)

#         # Optimization loop
#         for i in range(self.I):
#             # Sample J candidate action sequence
#             m = Normal(mean, std)
#             act_seq = m.sample_n(self.J)    # of shape (J, L, A)

#             # Copy and stack conditional frames, agent states, and action sequence
#             cond_frames = cond_frames.repeat(self.J, 1, 1, 1, 1)  # of shape (J, T, C, W, H)
#             cond_agstate = cond_agstate.repeat(self.J, 1, 1)      # of shape (J, T, S)
#             cond_actions = cond_actions.repeat(self.J, 1, 1)      # of shape (J, T, A)

#             # Concatenate conditional actions with future actions
#             act_total = torch.cat([cond_actions, act_seq], dim=1)   # of shape (J, T + L, A)

#             # Obtain rewards from predictive model
#             output = pred_model(cond_frames, cond_agstate, act_total)
#             rew_preds = output['rew_preds']     # of shape (J, L)
#             assert rew_preds.shape == (self.J, self.L)
#             ret_preds = rew_preds.sum(dim=1)    # Without discount

#             # Select top K action sequences based on cumulative rewards
#             _, idx = ret_preds.topk(self.K)
#             top_act_seq = torch.index_select(act_seq, dim=0, index=idx)     # of shape (K, L, A)

#             # Update parameters for normal distribution
#             std, mean = torch.std_mean(top_act_seq, dim=0)

#         # Return first action mean, of shape (A)
#         return mean[0, :]
