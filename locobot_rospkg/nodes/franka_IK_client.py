from eef_control.msg import (
    FrankaIKAction,
    FrankaIKGoal,
)
import actionlib
import rospy

PUSH_HEIGHT = 0.15
LOCO_FRANKA_DIFF = [-0.365,      -0.06103333]

def map_to_locobot_coord(franka_state):
    franka_state = np.copy(franka_state)
    franka_state[:2] += LOCO_FRANKA_DIFF
    return franka_state

def process_action(action, state):
    # TODO: figure out state bound for Franka
    # raise NotImplementedError()
    """
    If state + action is out of boundary, reverse action
    state: current eef position
    """
    output_action = np.copy(action)
    output_action = np.clip(output_action, -0.04, 0.04)
    # when input action drives the eef out of bound, revert it
    if len(state) < 2:
        print("Warning: input state has incorrect shape:", state)
        return output_action

    # convert franka state to locobot state bounds
    state = map_to_locobot_coord(state)
    end_pos = state[0:2] + action[:2]
    if end_pos[0] < 0.2 and end_pos[1] > -0.2 and end_pos[1] < 0.2:
        print("Warning: input action drives the eef self collision, revert it")
        output_action = -action
    if end_pos[1] > 0.44 - end_pos[0] or end_pos[1] < end_pos[0] - 0.44 \
            or end_pos[1] > end_pos[0] - 0.03 or end_pos[1] < -end_pos[0] + 0.03:
        print("Warning: input action drives the eef out of bound, revert it")
        output_action = -action
    return output_action


def preplan_trajectory(init_state, actions):
    curr_state = np.copy(init_state)
    way_points = []
    for t in range(actions.shape[0]):
        out_action = process_action(actions[t], curr_state)
        actions[t] = out_action
        curr_state = np.array([curr_state[0] + actions[t, 0],
                               curr_state[1] + actions[t, 1],
                               PUSH_HEIGHT])
        way_points.append(curr_state)
    way_points = np.stack(way_points)
    return way_points, actions


class FrankaIKClient(object):
    def __init__(self):
        self.client = actionlib.SimpleActionClient(
            'Franka_IK', FrankaIKAction)

    def send_ik_request(self, initial_qpos, batch_waypoints):
        self.client.wait_for_server(rospy.Duration(5))
        g = FrankaIKGoal()
        g.initial_qpos = initial_qpos
        g.num_traj, g.traj_length, g.waypoint_dim = batch_waypoints.shape
        g.way_points = batch_waypoints.reshape(-1)
        self.client.send_goal(g)
        self.client.wait_for_result(rospy.Duration(5))

        return self.client.get_result()

if __name__ == "__main__":
    import h5py
    import numpy as np

    rospy.init_node("Franka_IK_client")
    fik = FrankaIKClient()

    TRAJ_PATH = "data/data_2021-05-26_00:39:32.hdf5"
    with h5py.File(TRAJ_PATH, "r") as hf:
        gt_qpos = hf["qpos"][:]
        gt_eef_pos = hf["states"][:]
        actions = hf["actions"][:]
    # create waypoints
    way_points, actions = preplan_trajectory(gt_eef_pos[0], actions)

    batch_waypoints = np.array([way_points, way_points, way_points])
    result = fik.send_ik_request(gt_qpos[0], batch_waypoints)
    num_traj, traj_length, joint_dim = result.num_traj, result.traj_length, result.joint_dim
    qposes_pred = np.asarray(result.joint_angles).reshape(num_traj, traj_length, joint_dim)

    for gt_q, q in zip(gt_qpos, qposes_pred[0]):
        print(np.mean(gt_q - q, axis=0))
        print("=" * 80)