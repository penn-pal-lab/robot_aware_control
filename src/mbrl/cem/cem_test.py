import os

import numpy as np
import torch
from gym.envs.robotics import FetchPushEnv, FetchReachEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions.normal import Normal

# Hyperparameters
L = 3      # Prediction window size
I = 10      # Number of optimization iterations
J = 30     # Number of candidate action sequences
K = 5      # Number of top K candidate action sequences to select for optimization
num_episodes = 10  # Number of episodes to test


# Use actual gym environment to test cem algorithm
def cem_planner(env):
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

        # Generate rollout
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
        # mean = top_act_seq.mean(dim=0)
        # std = torch.sqrt(((top_act_seq - mean).pow(2)).sum(dim=0) / (K - 1))
        std, mean = torch.std_mean(top_act_seq, dim=0)

    # Print means of top returns, for debugging
    # print("\tMeans of top returns: ", ret_topks)
    # Return first action mean, of shape (A)
    return mean[0, :]

base_cls = FetchPushEnv
class FetchEnv(base_cls):
    def get_state(self):
        return self.sim.get_state()

    def set_state(self, env_state):
        assert env_state.qpos.shape == (self.sim.model.nq,) and env_state.qvel.shape == (self.sim.model.nv,)
        self.sim.set_state(env_state)
        self.sim.forward()


video_folder = os.path.join(os.path.dirname(__file__), "../../../videos")
os.makedirs(video_folder, exist_ok=True)
env = FetchEnv()
env.unwrapped.reward_type = 'dense'     # Set environment to dense rewards
# Do rollouts of CEM control
success_record = np.zeros(num_episodes)
for i in range(num_episodes):
    env.reset()
    vr = VideoRecorder(env, metadata=None, path=f'videos/test_{i}.mp4')

    ret = 0     # Episode return
    s = 0       # Step count
    print("\n=== Episode %d ===\n" % (i+1))
    while True:
        print("\tStep {}".format(s))
        action = cem_planner(env).numpy()       # Action convert to numpy array
        obs, rew, done, meta = env.step(action)
        ret += rew
        s += 1
        vr.capture_frame()

        # Calculate distance to goal at current step
        print("\tReward: {}".format(rew))

        if meta['is_success'] > 0 or done or s > 10:
            print("Episode %d --- total return: %.2f, is success: %f" % (i+1, ret, meta['is_success']))
            if meta['is_success'] > 0:
                success_record[i] = 1
            break
    vr.close()

# Close video recorder
env.close()

# Summary
print("\n\n### Summary ###")
print("\tSuccess rate: %.2f" % (success_record.sum() / len(success_record)))
