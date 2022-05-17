from src.utils.plot import save_gif
from src.dataset.robonet.robonet_dataset import create_heatmaps, denormalize, normalize
import torch
from src.env.robotics.masks.wx250s_mask_env import WX250sMaskEnv
import numpy as np
import torchvision.transforms as tf
import imageio
from src.utils.camera_calibration import camera_to_world_dict, LOCO_WX250S_DIFF


class WX250sAnalyticalModel(object):
    """
    Analytical model of the eef state and qpos of WX250s.
    """

    def __init__(self, config, bot, push_height, default_pitch, default_roll, cam_ext=None) -> None:
        super().__init__()
        self.bot = bot
        self._config = config
        # self.ik_solver = AIK()
        # self.env = LocobotMaskEnv()
        # self.env_thick = LocobotMaskEnv(thick=True)
        self.env = WX250sMaskEnv()
        self.env_thick = WX250sMaskEnv()
        if cam_ext is None:
            cam_ext = camera_to_world_dict[f"wx250s_c0"]
        self.env.set_opencv_camera_pose("main_cam", cam_ext)
        self.env_thick.set_opencv_camera_pose("main_cam", cam_ext)
        w, h = config.image_width, config.image_height
        self._img_transform = tf.Compose([tf.ToTensor(), tf.Resize((h, w))])
        self.push_height = push_height
        self.default_pitch = default_pitch
        self.default_roll = default_roll

    def run_inverse_kinematics(self, eef_curr, cur_arm_config=np.zeros(6)):
        """eef xyz -> qpos
        Args:
            eef_curr: (3, ) 3d position of eef
        """
        # self.bot.set_ee_pose_components(x=0, y=0, z=0, roll=0, pitch=0, yaw=None, custom_guess=None, execute=True, moving_time=None, accel_time=None, blocking=True)
        qpos = self.bot.arm.set_ee_pose_components(
            x=eef_curr[0],
            y=eef_curr[1],
            z=eef_curr[2],
            pitch=self.default_pitch,
            roll=self.default_roll,
            custom_guess=cur_arm_config,
            execute=False,
        )
        # qpos = np.zeros(5)
        # qpos[0:4] = self.ik_solver.ik(
        #     eef_curr, alpha=-self.default_pitch, cur_arm_config=cur_arm_config
        # )
        # qpos[4] = self.default_roll
        return qpos

    def predict_next_state_qpos(self, eef_curr, qpos_curr, action):
        """1-step robot state prediction: P(s' | s, a)
        eef_curr: (3, ) 3d position of eef
        qpos_curr: (6, )
        action: (2, ) planar action
        """

        eef_next = np.zeros(3)
        eef_next[0:2] = eef_curr[0:2] + action
        eef_next[2] = self.push_height

        # qpos_next = np.zeros(5)
        # qpos_next[0:4] = self.ik_solver.ik(
        #     eef_next, alpha=-self.default_pitch, cur_arm_config=qpos_curr[0:4]
        # )
        # qpos_next[4] = self.default_roll
        qpos_next = np.zeros(6)
        qpos_next, success = self.bot.arm.set_ee_pose_components(
            x=eef_next[0],
            y=eef_next[1],
            z=self.push_height,
            pitch=self.default_pitch,
            roll=self.default_roll,
            custom_guess=qpos_curr,
            execute=False,
        )
        return eef_next, qpos_next

    def predict_trajectory(self, eef_curr, qpos_curr, actions, thick=False):
        """
        Given the current pose of the robot and a list of K actions,
        predict the next K poses.

        Args:
            eef_curr (5,): xyz, rotation, gripper state
            qpos_curr (6, ): joint angles
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
        Assume state is in normalized locobot coordinate.
        Get the next timestep's states, masks, and heatmaps.
        """
        device = self._config.device
        T, B, S = data["states"].shape
        pred_states = torch.zeros_like(data["states"])
        w,h = self._config.image_width, self._config.image_height
        pred_masks = torch.zeros((T, B, 1, h, w), dtype=torch.float32, device=device)
        assert data["qpos"].shape[-1] == 6, f"wx250s qpos {data['qpos'].shape[-1]} != 6"
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

            # shift points from locobot to wx250s frame.
            # loco - (loco - wx250s) = wx250s
            start_state[:2] -= LOCO_WX250S_DIFF
            raw_p_states, p_masks = self.predict_trajectory(
                start_state, start_qpos, actions, thick=thick
            )
            # shift points back to locobot frame.
            # franka + (loco - franka) = loco
            raw_p_states[:, :2] += LOCO_WX250S_DIFF
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

        return pred_states, pred_masks