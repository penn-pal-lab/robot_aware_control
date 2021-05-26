from eef_control.msg import (
    PoseControlAction,
    PoseControlGoal,
)
import actionlib
import rospy

PUSH_HEIGHT = 0.15
START_POSE = [0.55, 0, PUSH_HEIGHT, 0, 1, 0, 0]


class FrankaControlClient(object):
    def __init__(self):
        self.client = actionlib.SimpleActionClient(
            'Franka_Control', PoseControlAction)

    def send_target_eef_request(self, target_pose):
        self.client.wait_for_server(rospy.Duration(5))
        g = PoseControlGoal()
        g.target_pose = target_pose
        self.client.send_goal(g)
        self.client.wait_for_result(rospy.Duration(5))

        return self.client.get_result()

if __name__ == "__main__":
    rospy.init_node("Franka_Control_client")
    fik = FrankaControlClient()

    fik.send_target_eef_request(START_POSE)