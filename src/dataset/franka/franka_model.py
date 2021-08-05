from locobot_rospkg.nodes.franka_IK_client import FrankaIKClient
from src.utils.plot import save_gif
from src.dataset.robonet.robonet_dataset import create_heatmaps, denormalize, normalize
import torch
from src.env.robotics.masks.franka_mask_env import FrankaMaskEnv
import numpy as np
import torchvision.transforms as tf
import imageio
from src.utils.camera_calibration import camera_to_world_dict, LOCO_FRANKA_DIFF


PUSH_HEIGHT = 0.12

class FrankaAnalyticalModel(object):
    """
    Analytical model of the eef state and qpos of locobot.
    """

    def __init__(self, config, franka_ik, cam_ext=None) -> None:
        super().__init__()
        self._config = config
        self.ik_solver:FrankaIKClient = franka_ik
        self.env = FrankaMaskEnv()
        if cam_ext is None:
            cam_ext = camera_to_world_dict[f"franka_c0"]
        self.env.set_opencv_camera_pose("main_cam", cam_ext)
        w, h = config.image_width, config.image_height
        self._img_transform = tf.Compose([tf.ToTensor(), tf.Resize((h, w))])

    def predict_batch(self, data, thick=False):
        """
        Assume the state is in normalized locobot coordinate?
        Get the next timestep's states, masks, and heatmaps.
        """
        device = self._config.device
        T, B, S = data["states"].shape
        pred_states = torch.zeros_like(data["states"])
        w,h = self._config.image_width, self._config.image_height
        pred_masks = torch.zeros((T, B, 1, h, w), dtype=torch.float32, device=device)
        assert data["qpos"].shape[-1] == 7, f"franka  qpos {data['qpos'].shape[-1]} != 7"

        # for i in range(B):
        start_qpos = data["qpos"][0].cpu().numpy()
        actions = data["actions"][:].cpu().numpy()
        low = data["low"].cpu()
        high = data["high"].cpu()

        waypoints = denormalize(data["states"], low, high).numpy()
        # shift points from locobot to franka frame.
        # loco - (loco - franka) = franka
        waypoints[:,:,:2] -= LOCO_FRANKA_DIFF

        # create waypoints from actions
        for t in range(actions.shape[0]):
            act = actions[t,:, :3]
            waypoints[t+1, :, :3] = waypoints[t,:, :3] + act
        # make waypoints B x T x 3
        waypoints = waypoints.transpose(1,0,2)
        # ASSUME start_qpos is the same qpos repeated across batch dimension!
        result = self.ik_solver.send_ik_request(start_qpos[0], waypoints)
        num_traj, traj_length, joint_dim = result.num_traj, result.traj_length, result.joint_dim
        # qposes is B x T x 7
        qposes_pred = np.asarray(result.joint_angles).reshape(num_traj, traj_length, joint_dim)

        for i, qpose_traj in enumerate(qposes_pred):
            masks = self.env.generate_masks(qpose_traj)
            masks = (
                torch.stack([self._img_transform(i) for i in masks])
                .type(torch.bool)
                .type(torch.float32)
            )
            pred_masks[:, i] = masks.to(device, non_blocking=True)

        # shift points back to locobot frame.
        # franka + (loco - franka) = loco
        waypoints[:, :, :2] += LOCO_FRANKA_DIFF
        waypoints = torch.from_numpy(waypoints.transpose(1,0,2)) # T x B x 5
        # normalize the states again
        p_states = normalize(waypoints, low, high)
        pred_states = p_states.to(device, non_blocking=True)

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