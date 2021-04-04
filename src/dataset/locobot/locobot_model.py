from src.dataset.robonet.robonet_dataset import denormalize, normalize
import torch
from src.env.robotics.masks.locobot_mask_env import LocobotMaskEnv
import numpy as np
from src.env.robotics.masks.locobot_analytical_ik import (
    AnalyticInverseKinematics as AIK,
)
import torchvision.transforms as tf
import imageio
from src.utils.camera_calibration import camera_to_world_dict


# TODO: record pitch/roll in eef pose in the future
PUSH_HEIGHT = 0.15
DEFAULT_PITCH = 1.3
DEFAULT_ROLL = 0.0


class LocobotAnalyticalModel(object):
    """
    Analytical model of the eef state and qpos of locobot.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self._config = config
        self.ik_solver = AIK()
        self.env = LocobotMaskEnv()
        cam_ext = camera_to_world_dict[f"locobot_c0"]
        self.env.set_opencv_camera_pose("main_cam", cam_ext)
        w, h = config.image_width, config.image_height
        self._img_transform = tf.Compose([tf.ToTensor(), tf.Resize((h, w))])

    def run_inverse_kinematics(self, eef_curr, cur_arm_config=np.zeros(4)):
        """eef xyz -> qpos
        Args:
            eef_curr: (3, ) 3d position of eef
        """
        qpos = np.zeros(5)
        qpos[0:4] = self.ik_solver.ik(
            eef_curr, alpha=-DEFAULT_PITCH, cur_arm_config=cur_arm_config
        )
        qpos[4] = DEFAULT_ROLL
        return qpos

    def predict_next_state_qpos(self, eef_curr, qpos_curr, action):
        """1-step robot state prediction: P(s' | s, a)
        eef_curr: (3, ) 3d position of eef
        qpos_curr: (5, )
        action: (2, ) planar action
        """

        eef_next = np.zeros(3)
        eef_next[0:2] = eef_curr[0:2] + action
        eef_next[2] = PUSH_HEIGHT

        qpos_next = np.zeros(5)
        qpos_next[0:4] = self.ik_solver.ik(
            eef_next, alpha=-DEFAULT_PITCH, cur_arm_config=qpos_curr[0:4]
        )
        qpos_next[4] = DEFAULT_ROLL
        return eef_next, qpos_next

    def predict_trajectory(self, eef_curr, qpos_curr, actions):
        """
        Given the current pose of the robot and a list of K actions,
        predict the next K poses.

        Args:
            eef_curr (5,): xyz, rotation, gripper state
            qpos_curr (5, ): joint angles
            actions (3, ): xyz displacement

        Returns:
            states, masks, heatmaps, etc. of the next K timesteps
        """
        start_eef = eef_curr
        start_qpos = qpos_curr
        states = [eef_curr]
        pred_qpos = [qpos_curr]
        for t in range(len(actions)):
            # closed loop prediction
            act = actions[t, :2]
            eef_curr, qpos_curr = self.predict_next_state_qpos(eef_curr, qpos_curr, act)
            # open loop prediction
            # act = np.sum(actions[0:t+1, :2], 0)
            # eef_curr, qpos_curr = self.predict_next_state_qpos(start_eef, start_qpos, act)
            # add rotation and gripper state as 0
            eef_curr = np.concatenate([eef_curr, [0,0]])
            states.append(eef_curr)
            pred_qpos.append(qpos_curr)

        pred_qpos = np.stack(pred_qpos)
        masks = self.env.generate_masks(pred_qpos)
        masks = (
            torch.stack([self._img_transform(i) for i in masks])
            .type(torch.bool)
            .type(torch.float32)
        )
        states = torch.from_numpy(np.stack(states).astype(np.float32))
        return states, masks

    def predict_batch(self, data):
        device = self._config.device
        B = data["states"].shape[1]
        pred_states = torch.zeros_like(data["states"])
        pred_masks = torch.zeros_like(data["masks"])
        for i in range(B):
            start_state = data["states"][0, i].cpu().numpy()
            low = data["low"][i].cpu()
            high = data["high"][i].cpu()
            start_state = denormalize(start_state, low.numpy(), high.numpy())
            start_qpos = data["qpos"][0, i].cpu().numpy()
            if self._config.preprocess_action != "raw":
                actions = data["raw_actions"][:, i].cpu().numpy()
            else:
                actions = data["actions"][:, i].cpu().numpy()

            p_states, p_masks = self.predict_trajectory(
                start_state, start_qpos, actions
            )
            # normalize the states again
            p_states = normalize(p_states, low, high)
            pred_states[:, i] = p_states.to(device, non_blocking=True)
            pred_masks[:, i] = p_masks.to(device, non_blocking=True)

            # visualize the projected masks
            # diff = p_masks.cpu().type(torch.bool) ^ data["masks"][:, i].cpu().type(torch.bool)
            # diff= (255 * diff.cpu().squeeze().numpy()).astype(np.uint8)
            # p_masks = (255 * p_masks.cpu().squeeze().numpy()).astype(np.uint8)
            # masks = (255 * data["masks"][:, i].cpu().squeeze().numpy()).astype(np.uint8)
            # gif = np.concatenate([masks, p_masks, diff], 2)
            # imageio.mimwrite(f"{i}_mask.gif", gif)
            # import ipdb; ipdb.set_trace()
        return pred_states, pred_masks