from src.utils.plot import save_gif
from src.dataset.robonet.robonet_dataset import create_heatmaps, denormalize, normalize
import torch
from src.env.robotics.masks.locobot_mask_env import LocobotMaskEnv
import numpy as np
from src.env.robotics.masks.locobot_analytical_ik import (
    AnalyticInverseKinematics as AIK,
)
import torchvision.transforms as tf
import imageio
from src.utils.camera_calibration import camera_to_world_dict, world_to_camera_dict


# TODO: record pitch/roll in eef pose in the future
PUSH_HEIGHT = 0.15
DEFAULT_PITCH = 1.3
DEFAULT_ROLL = 0.0


class LocobotAnalyticalModel(object):
    """
    Analytical model of the eef state and qpos of locobot.
    """

    def __init__(self, config, cam_ext=None) -> None:
        super().__init__()
        self._config = config
        self.ik_solver = AIK()
        self.env = LocobotMaskEnv()
        self.env_thick = LocobotMaskEnv(thick=True)
        if cam_ext is None:
            cam_ext = camera_to_world_dict[f"locobot_modified_c0"]
        self.env.set_opencv_camera_pose("main_cam", cam_ext)
        self.env_thick.set_opencv_camera_pose("main_cam", cam_ext)
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

    def predict_trajectory(self, eef_curr, qpos_curr, actions, thick=False):
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
        states = [eef_curr]
        pred_qpos = [qpos_curr]
        for t in range(len(actions)):
            act = actions[t, :2]
            eef_curr, qpos_curr = self.predict_next_state_qpos(eef_curr, qpos_curr, act)
            # add rotation and gripper state as 0
            eef_curr = np.concatenate([eef_curr, [0,0]])
            states.append(eef_curr)
            pred_qpos.append(qpos_curr)

        pred_qpos = np.stack(pred_qpos)
        if thick:
            masks = self.env_thick.generate_masks(pred_qpos)
        else:
            masks = self.env.generate_masks(pred_qpos)
        masks = (
            torch.stack([self._img_transform(i) for i in masks])
            .type(torch.bool)
            .type(torch.float32)
        )
        states = torch.from_numpy(np.stack(states).astype(np.float32))
        return states, masks

    def predict_batch(self, data, thick=False):
        """
        Get the next timestep's states, masks, and heatmaps.
        """
        device = self._config.device
        T, B, S = data["states"].shape
        pred_states = torch.zeros_like(data["states"])
        w,h = self._config.image_width, self._config.image_height
        pred_masks = torch.zeros((T, B, 1, h, w), dtype=torch.float32, device=device)
        assert data["qpos"].shape[-1] == 5, f"locobot qpos {data['qpos'].shape[-1]} != 5"
        for i in range(B):
            start_qpos = data["qpos"][0, i].cpu().numpy()
            if self._config.preprocess_action != "raw":
                actions = data["raw_actions"][:, i].cpu().numpy()
                # normalized world states
                low = data["raw_low"][i].cpu()
                high = data["raw_high"][i].cpu()
                c_low = data["low"][i].cpu()
                c_high = data["high"][i].cpu()
                start_state = data["raw_states"][0, i].cpu().numpy()
                start_state = denormalize(start_state, low.numpy(), high.numpy())
            else:
                actions = data["actions"][:, i].cpu().numpy()
                low = data["low"][i].cpu()
                high = data["high"][i].cpu()
                start_state = data["states"][0, i].cpu().numpy()
                start_state = denormalize(start_state, low.numpy(), high.numpy())

            raw_p_states, p_masks = self.predict_trajectory(
                start_state, start_qpos, actions, thick=thick
            )
            # normalize the states again
            p_states = normalize(raw_p_states, low, high)
            pred_states[:, i] = p_states.to(device, non_blocking=True)
            pred_masks[:, i] = p_masks.to(device, non_blocking=True)

            # >>>>>>>> visualize the projected masks
            # diff = p_masks.cpu().type(torch.bool) ^ data["masks"][:, i].cpu().type(torch.bool)
            # diff= (255 * diff.cpu().squeeze().numpy()).astype(np.uint8)
            # p_masks = (255 * p_masks.cpu().squeeze().numpy()).astype(np.uint8)
            # masks = (255 * data["masks"][:, i].cpu().squeeze().numpy()).astype(np.uint8)
            # gif = np.concatenate([masks, p_masks, diff], 2)
            # imageio.mimwrite(f"{i}_mask.gif", gif)
            # import ipdb; ipdb.set_trace()

        # >>>>>>>> compute the average error per timestep
        # raw_states = denormalize(data["raw_states"].cpu(), data["raw_low"], data["raw_high"])
        # p_states = denormalize(pred_states.cpu(), data["raw_low"], data["raw_high"])
        # diff = (p_states.cpu() - raw_states).abs()[:, :, :3]
        # diff= diff.mean(1)
        # print("raw diff")
        # print(diff)
        ''' convert pred_states to camera space if necessary'''
        if "camera" in self._config.preprocess_action:
            # flatten into array
            raw_pred_eef = pred_states[:, :, :3].flatten(0,1).cpu() # (T, B, 5)
            raw_pred_eef = denormalize(raw_pred_eef, low[:3], high[:3]).numpy()
            raw_pred_eef = np.concatenate([raw_pred_eef, np.ones((T*B, 1))], 1).T
            # denormalize
            # TODO: account for multiple viewpoints of locobot
            world2cam = world_to_camera_dict["locobot_c0"]
            c_pred_eef = (world2cam @ raw_pred_eef).T[:, :3] # (T*B, 3)
            c_pred_eef = normalize(torch.from_numpy(c_pred_eef), c_low[:3], c_high[:3])
            c_pred_eef = c_pred_eef.reshape(T,B,3)
            # normalize in camera space
            pred_states[:, :, :3] = c_pred_eef
            # >>>>>>>> compute the average error per timestep
            # states = denormalize(data["states"].cpu(), data["low"], data["high"])
            # p_states = denormalize(pred_states.cpu(), data["low"], data["high"])
            # diff = (p_states.cpu() - states).abs()[:, :, :3]
            # diff= diff.mean(1)
            # print("cam diff")
            # print(diff)
            # import ipdb; ipdb.set_trace()

        if self._config.model_use_heatmap:
            # TODO: account for camera space states
            # T x B x 1 x H x W
            heatmaps = data["heatmaps"].clone()
            for idx in range(heatmaps.shape[1]):
                # get states from t=1:T to generate heatmas
                s = pred_states[1:, idx].cpu()
                low = data["low"][idx].cpu().numpy().squeeze()
                high = data["high"][idx].cpu().numpy().squeeze()
                robot = data["robot"][idx]
                vp = data["folder"][idx]
                # import ipdb; ipdb.set_trace()
                hm = create_heatmaps(s, low, high, robot, vp)
                # use gt heatmap at t=0
                heatmaps[1:, idx] = torch.from_numpy(hm)

            # >>>>>>>> visualize heatmaps
            # images = data["images"]
            # hm = heatmaps.repeat(1,1,3,1,1)
            # hm_images = (images * hm).transpose(0,1).unsqueeze(2)
            # gt_hm = data["heatmaps"].repeat(1,1,3,1,1)
            # gt_hm_images = (images * gt_hm).transpose(0,1).unsqueeze(2)
            # gif = torch.cat([gt_hm_images, hm_images], 2)
            # save_gif("hm.gif", gif)
            # import ipdb; ipdb.set_trace()
            return pred_states, pred_masks, heatmaps

        return pred_states, pred_masks