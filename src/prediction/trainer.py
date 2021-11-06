import logging
import os
from collections import defaultdict
from functools import partial

import torchvision.transforms as tf
from src.dataset.locobot.locobot_model import LocobotAnalyticalModel
from src.dataset.robonet.robonet_dataset import (
    create_heatmaps,
    get_batch,
    process_batch,
)
from src.prediction.models.dynamics import (
    CopyModel,
    DeterministicConvModel,
    GripperStatePredictor,
    JointPosPredictor,
    SVGConvModel,
    SVGModel,
)
from src.utils.camera_calibration import camera_to_world_dict
from src.utils.image import zero_robot_region

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from warnings import simplefilter  # disable tensorflow warnings

simplefilter(action="ignore", category=FutureWarning)

from math import floor
from time import time

import colorlog
import imageio
import ipdb
import numpy as np
import torch
import wandb
from src.prediction.losses import (
    dontcare_l1_criterion,
    dontcare_mse_criterion,
    kl_criterion,
    l1_criterion,
    mse_criterion,
    robot_mse_criterion,
    world_mse_criterion,
    world_psnr_criterion,
)
from src.utils.metrics import psnr, ssim
from src.utils.plot import save_gif, save_gif_with_text
from torch import optim
from tqdm import tqdm


class PredictionTrainer(object):
    """
    Video Prediction with multiple robot dataset
    Training, Checkpointing, Visualizing the prediction
    """

    def __init__(self, config):
        self._config = config
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f"using {device} for training")
        self._logger = colorlog.getLogger("file/console")

        self._device = config.device = device
        self._init_models(config)
        self._scheduled_sampling = config.scheduled_sampling

        # init WandB
        if not config.wandb:
            os.environ["WANDB_MODE"] = "dryrun"
        else:
            assert "WANDB_API_KEY" in os.environ, "set WANDB_API_KEY env var."
        wandb.init(
            resume=config.jobname,
            project=config.wandb_project,
            config=config,
            dir=config.log_dir,
            entity=config.wandb_entity,
            group=config.wandb_group,
            job_type=config.wandb_job_type,
            config_exclude_keys=["device"],
        )
        self._img_augmentation = config.img_augmentation
        self._plot_rng = np.random.RandomState(self._config.seed)
        w, h = config.image_width, config.image_height
        self._img_transform = tf.Compose([tf.ToTensor(), tf.Resize((h, w))])
        self._video_sample_rng = np.random.RandomState(self._config.seed)

    def _init_models(self, cf):
        """Initialize models and optimizers
        When adding a new model, make sure to:
        - Call to(device)
        - Add optimizer
        - Add optimizer step() call
        - Update save and load ckpt code
        """
        if cf.model == "svg":
            # self.model = SVGConvModel(cf).to(self._device)
            self.model = SVGModel(cf).to(self._device)
        elif cf.model == "det":
            self.model = DeterministicConvModel(cf).to(self._device)
        elif cf.model == "copy":
            self.model = CopyModel()
            return
        else:
            raise ValueError(f"{cf.model}")

        if cf.optimizer == "adam":
            optimizer = partial(optim.Adam, lr=cf.lr, betas=(cf.beta1, 0.999))
        elif cf.optimizer == "rmsprop":
            optimizer = optim.RMSprop
        elif cf.optimizer == "sgd":
            optimizer = optim.SGD
        else:
            raise ValueError("Unknown optimizer: %s" % cf.optimizer)

        # self.background_model = Attention(cf).to(self._device)
        # self.background_model.apply(init_weights)
        # params = list(self.model.parameters()) + list(self.background_model.parameters())
        params = list(self.model.parameters())
        self.optimizer = optimizer(params)
        if "finetune" in cf.experiment:
            if cf.experiment == "finetune_locobot":
                self.robot_model = LocobotAnalyticalModel(cf)
            elif cf.learned_robot_model:
                # learned robot models for evaluation
                self.joint_model = JointPosPredictor(cf).to(self._device)
                self.gripper_model = GripperStatePredictor(cf).to(self._device)
                self._load_robot_model_checkpoint(cf.robot_model_ckpt)

    def _schedule_prob(self):
        """Returns probability of using ground truth"""
        # assume 50k max training steps
        # https://www.desmos.com/calculator/czolgrcelz
        k = self._config.scheduled_sampling_k
        use_truth = k / (k + np.exp(self._step / k))
        use_model = 1 - use_truth
        return [use_truth, use_model]

    def _use_true_token(self):
        """
        Scheduled Sampling: Decide whether to use model output or ground truth
        """
        if not self._scheduled_sampling:
            return True
        return np.random.choice([True, False], p=self._schedule_prob())

    def _recon_loss(self, prediction, target, mask=None, batch_weight=None):
        if self._config.reconstruction_loss == "mse":
            return mse_criterion(prediction, target)
        elif self._config.reconstruction_loss == "l1":
            return l1_criterion(prediction, target, batch_weight)
        elif self._config.reconstruction_loss == "dontcare_mse":
            robot_weight = self._config.robot_pixel_weight
            return dontcare_mse_criterion(prediction, target, mask, robot_weight)
        elif self._config.reconstruction_loss == "dontcare_l1":
            robot_weight = self._config.robot_pixel_weight
            return dontcare_l1_criterion(prediction, target, mask, robot_weight, batch_weight)
        else:
            raise NotImplementedError(f"{self._config.reconstruction_loss}")


    @torch.no_grad()
    def _generate_learned_robot_states(self, data):
        """
        Use the learned robot model to generate masks and states for
        training / eval.
        """
        cf = self._config
        states = data["states"]
        ac = data["actions"]
        if self._config.preprocess_action != "raw":
            ac = data["raw_actions"]
        mask = data["masks"]
        qpos = data["qpos"]
        viewpoints = set(data["folder"])
        if not hasattr(self, "renderers"):
            self.renderers = {}
        for v in viewpoints:
            if v in self.renderers:
                continue
            if cf.experiment == "finetune":
                from src.env.robotics.masks.baxter_mask_env import BaxterMaskEnv

                env = BaxterMaskEnv()
                env.arm = "left"
                cam_ext = camera_to_world_dict[f"baxter_{v}"]
            elif cf.experiment == "finetune_widowx":
                from src.env.robotics.masks.widowx_mask_env import WidowXMaskEnv

                env = WidowXMaskEnv()
                cam_ext = camera_to_world_dict[f"widowx_{v}"]
            elif cf.experiment == "finetune_sawyer_view":
                from src.env.robotics.masks.sawyer_mask_env import SawyerMaskEnv

                env = SawyerMaskEnv()
                cam_ext = camera_to_world_dict[f"sawyer_{v}"]
            else:
                raise ValueError

            env.set_opencv_camera_pose("main_cam", cam_ext)
            self.renderers[v] = env

        # use robot models to generate mask / eef pose instead of gt
        predicted_states = torch.zeros_like(states)
        predicted_states[0] = states[0]
        predicted_masks = torch.zeros_like(mask)
        predicted_masks[0] = mask[0]
        q_j, r_j = qpos[0], states[0]
        for i in range(1, len(ac) + 1):
            a_j = ac[i - 1]
            r_pred = self.gripper_model(r_j, a_j) + r_j
            q_pred = self.joint_model(q_j, a_j) + q_j
            predicted_states[i] = r_pred
            # generate mask for each qpos prediction
            for b in range(q_pred.shape[0]):
                vp = data["folder"][b]
                q_pred_b = q_pred[b].cpu().numpy()
                env = self.renderers[vp]
                m = env.generate_masks([q_pred_b])[0]
                m = (
                    self._img_transform(m)
                    .to(self._device, non_blocking=True)
                    .type(torch.bool)
                )
                predicted_masks[i][b] = m
            q_j = q_pred
            r_j = r_pred

        states = predicted_states
        mask = predicted_masks
        # TODO: transform the state back to camera space
        if self._config.model_use_heatmap:
            # T x B x 1 x H x W
            heatmaps = data["heatmaps"].clone()
            for idx in range(heatmaps.shape[1]):
                # get states from t=1:T to generate heatmas
                s = states[1:, idx].cpu()
                low = data["low"][idx].cpu().numpy().squeeze()
                high = data["high"][idx].cpu().numpy().squeeze()
                robot = data["robot"][idx]
                vp = data["folder"][idx]
                hm = create_heatmaps(s, low, high, robot, vp)
                # use gt heatmap at t=0
                heatmaps[1:, idx] = torch.from_numpy(hm)
            # images = data["images"]
            # hm = heatmaps.repeat(1,1,3,1,1)
            # hm_images = (images * hm).transpose(0,1).unsqueeze(2)
            # gt_hm = data["heatmaps"].repeat(1,1,3,1,1)
            # gt_hm_images = (images * gt_hm).transpose(0,1).unsqueeze(2)
            # gif = torch.cat([gt_hm_images, hm_images], 2)
            # save_gif("hm.gif", gif)
            # ipdb.set_trace()

            return states, mask, heatmaps
        return states, mask

    def _train_video(self, data):
        """Train the model over the video data

        Slices video up into K length sequences for training.
        Args:
            data (dict): Video data
        """
        cf = self._config
        x = data["images"]
        T = len(x)
        window = cf.n_past + cf.n_future
        self.steps_per_train_video = floor(T / window)
        all_losses = defaultdict(float)

        for i in range(floor(T / window)):
            if cf.random_snippet:
                s = self._video_sample_rng.randint(0, (T - window) + 1)
                e = s + window
            else:
                s = i * window
                e = (i + 1) * window
            batch_data = {
                "images": x[s:e],
                "states": data["states"][s:e],
                "actions": data["actions"][s : e - 1],
                "masks": data["masks"][s:e],
                "qpos": data["qpos"][s:e],
                "robot": data["robot"],
                "folder": data["folder"],
            }
            if cf.model_use_heatmap:
                batch_data["heatmaps"] = data["heatmaps"][s:e]
            if cf.load_movement_info:
                batch_data["high_movement"] = data["high_movement"]

            if "finetune" in cf.experiment and (cf.model_use_mask or cf.model_use_robot_state):
                if cf.preprocess_action != "raw":
                    batch_data["raw_actions"] = data["raw_actions"][s : e - 1]
                    batch_data["raw_states"] = data["raw_states"][s : e]
                    batch_data["raw_low"] = data["raw_low"]
                    batch_data["raw_high"] = data["raw_high"]

                batch_data["low"] = data["low"]
                batch_data["high"] = data["high"]

                if cf.experiment == "finetune_locobot":
                    # use analytical model
                    # inputs: current qpos, state, list of actions
                    # outputs: list of states, list of masks
                    out = self.robot_model.predict_batch(batch_data)

                elif cf.learned_robot_model:
                    out = self._generate_learned_robot_states(batch_data)

                if cf.model_use_heatmap:
                    states, masks, heatmaps = out
                    batch_data["heatmaps"] = heatmaps
                else:
                    states, masks = out
                batch_data["states"] = states
                batch_data["masks"] = masks

            losses = self._train_step(batch_data)
            for k, v in losses.items():
                all_losses[k] += v / floor(T / window)
        return all_losses

    def _train_step(self, data):
        """Forward and Backward pass of models
        Returns info dict containing loss metrics
        """
        cf = self._config
        losses = defaultdict(float)  # log loss metrics
        recon_loss = kld = 0
        x = data["images"]
        states = data["states"]
        ac = data["actions"]
        mask = data["masks"]
        if cf.load_movement_info:
            movement_info = data["high_movement"]
        if cf.model_use_heatmap:
            heatmaps = data["heatmaps"]
        robot_name = data["robot"]
        robot_name = np.array(robot_name)
        all_robots = set(robot_name)
        x_pred = None
        skip = None

        self.model.zero_grad()
        bs = min(cf.batch_size, x.shape[1])
        self.model.init_hidden(bs)  # initialize the recurrent states

        # background mask
        for i in range(1, cf.n_past + cf.n_future):
            if i > 1:
                x_j = x[i - 1] if self._use_true_token() else x_pred.clone()
            else:
                x_j = x[i - 1]
            # let j be i - 1, or previous timestep
            m_j, r_j, a_j = mask[i - 1], states[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], mask[i], states[i]
            hm_j, hm_i = None, None
            if cf.model_use_heatmap:
                hm_j, hm_i = heatmaps[i - 1], heatmaps[i]

            # zero out robot pixels in input for norobot cost
            x_j_black, x_i_black = x_j, x_i
            if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                x_j_black = zero_robot_region(m_j, x_j)
                x_i_black = zero_robot_region(m_i, x_i)

            if cf.last_frame_skip:  # always use skip of current img
                skip = None

            m_in = m_j
            if cf.model_use_future_mask:
                m_in = torch.cat([m_j, m_i], 1)
            r_in = r_j
            if cf.model_use_future_robot_state:
                r_in = (r_j, r_i)
            hm_in = hm_j
            if cf.model_use_future_heatmap:
                hm_in = torch.cat([hm_j, hm_i], 1)

            if cf.model == "det":
                x_pred, curr_skip = self.model(x_j_black, m_in, r_j, a_j, skip)
            elif cf.model == "svg":
                m_next_in = m_i
                if cf.model_use_future_mask:
                    m_next_in = m_i.repeat(1, 2, 1, 1)
                hm_next_in = hm_i
                if cf.model_use_future_heatmap:
                    hm_next_in = hm_i.repeat(1, 2, 1, 1)
                # out = self.model(
                #     x_j_black,
                #     m_in,
                #     r_in,
                #     hm_in,
                #     a_j,
                #     x_i_black,
                #     m_next_in,
                #     r_i,
                #     hm_next_in,
                #     skip,
                # )
                # SVG latent cost version
                out = self.model(
                    x_j_black,
                    m_in,
                    r_in,
                    a_j,
                    x_i_black,
                    m_next_in,
                    r_i,
                    hm_next_in,
                    skip,
                )
                x_pred, curr_skip, mu, logvar, mu_p, logvar_p = out

            x_pred, x_pred_mask = x_pred[:, :3], x_pred[:, 3].unsqueeze(1)
            x_pred = (1 - x_pred_mask) * x_j + (x_pred_mask) * x_pred

            if i <= cf.n_past:
                # overwrite skip with most recent conditioning frame skip
                skip = curr_skip
            # calculate loss per view and log it
            if cf.multiview:
                num_views = x_pred.shape[2] // cf.image_width
                for n in range(num_views):
                    start, end = n * cf.image_width, (n + 1) * cf.image_width
                    view_pred = x_pred[:, :, start:end, :]
                    view = x[i][:, :, start:end, :]
                    view_mask = mask[i][:, :, start:end, :]
                    view_loss = self._recon_loss(view_pred, view, view_mask)
                    recon_loss += view_loss
                    view_loss_scalar = view_loss.cpu().item()
                    losses[f"view_{n}"] += view_loss_scalar
                    losses["recon_loss"] += view_loss_scalar
            else:
                if self._config.load_movement_info:
                    batch_weight = (self._config.movement_weight * movement_info).to(self._device)
                    batch_weight[~movement_info] = 1.0
                    view_loss = self._recon_loss(x_pred, x_i, m_i, batch_weight)
                else:
                    view_loss = self._recon_loss(x_pred, x_i, m_i)
                recon_loss += view_loss  # accumulate for backprop
                losses["recon_loss"] += view_loss.cpu().item()

                # for logging metrics
                with torch.no_grad():
                    robot_mse = robot_mse_criterion(x_pred, x_i, m_i)
                    world_mse = world_mse_criterion(x_pred, x_i, m_i)
                    losses["robot_loss"] += robot_mse.cpu().item()
                    losses["world_loss"] += world_mse.cpu().item()
                    # robot specific metrics
                    for r in all_robots:
                        if len(all_robots) == 1:
                            break
                        r_idx = r == robot_name
                        r_pred = x_pred[r_idx]
                        r_img = x_i[r_idx]
                        r_mask = m_i[r_idx]
                        r_robot_mse = robot_mse_criterion(r_pred, r_img, r_mask)
                        r_world_mse = world_mse_criterion(r_pred, r_img, r_mask)
                        losses[f"{r}_robot_loss"] += r_robot_mse.cpu().item()
                        losses[f"{r}_world_loss"] += r_world_mse.cpu().item()

            if cf.model == "svg":
                bs = min(cf.batch_size, x.shape[1])
                kl = kl_criterion(mu, logvar, mu_p, logvar_p, bs)
                kld += kl
                losses["kld"] += kl.cpu().item()
        loss = recon_loss + kld * cf.beta
        loss.backward()
        self.optimizer.step()

        for k, v in losses.items():
            losses[k] = v / cf.n_future
        return losses

    def _compute_epoch_metrics(self, data_loader, name):
        """Compute the metrics over an entire epoch of data

        Args:
            data_loader (Dataloader): data loader for the dataset
            name (str): name of the dataset
        """
        losses = defaultdict(list)
        progress = tqdm(total=len(data_loader), desc=f"computing {name} epoch")
        for data in data_loader:
            data = process_batch(data, self._device)
            info = self._eval_video(data, autoregressive=True)
            for k, v in info.items():
                losses[k].append(v)
            progress.update()

        avg_loss = {f"{name}/{k}": np.mean(v) for k, v in losses.items()}
        # epoch_loss = {f"test/epoch_{k}": np.sum(v) for k, v in losses.items()}
        log_str = ""
        for k, v in avg_loss.items():
            log_str += f"{k}: {v:.5f}, "
        self._logger.info(log_str)
        return avg_loss

    def _eval_video(self, data, autoregressive=False):
        """Evaluates over an entire video
        data: video data from dataloader
        autoregressive: use model's outputs as input for next timestep
        """
        cf = self._config
        num_samples = 1
        if autoregressive and cf.model == "svg" and "finetune" in cf.experiment:
            num_samples = 3
        x = data["images"]
        T = len(x)
        window = cf.n_eval
        sampled_losses = [
            defaultdict(float) for _ in range(num_samples)
        ]  # list of video sample losses
        for i in range(floor(T / window)):
            s = i * window
            e = (i + 1) * window
            batch_data = {
                "images": x[s:e],
                "states": data["states"][s:e],
                "actions": data["actions"][s : e - 1],
                "masks": data["masks"][s:e],
                "robot": data["robot"],
                "qpos": data["qpos"][s:e],
                "folder": data["folder"],
                "file_path": data["file_path"],
            }
            if cf.model_use_heatmap:
                batch_data["heatmaps"] = data["heatmaps"][s:e]

            if "finetune" in cf.experiment and (cf.model_use_mask or cf.model_use_robot_state):
                if cf.preprocess_action != "raw":
                    batch_data["raw_actions"] = data["raw_actions"][s : e - 1]
                    batch_data["raw_states"] = data["raw_states"][s : e]
                    batch_data["raw_low"] = data["raw_low"]
                    batch_data["raw_high"] = data["raw_high"]

                batch_data["low"] = data["low"]
                batch_data["high"] = data["high"]

                if cf.experiment == "finetune_locobot":
                    # use analytical model
                    # inputs: current qpos, state, list of actions
                    # outputs: list of states, list of masks
                    out = self.robot_model.predict_batch(batch_data)
                elif cf.learned_robot_model:
                    out = self._generate_learned_robot_states(batch_data)

                if cf.model_use_heatmap:
                    states, masks, heatmaps = out
                    batch_data["heatmaps"] = heatmaps
                else:
                    states, masks = out

                batch_data["states"] = states
                # eval also needs true masks for computing world errors
                batch_data["pred_masks"] = masks
            else:  # use true masks for eval rollout
                batch_data["pred_masks"] = batch_data["masks"]

            for sample in range(num_samples):
                losses = self._eval_step(batch_data, autoregressive)
                for k, v in losses.items():
                    sampled_losses[sample][k] += v

        # now pick the best sample by world error, and average over frames
        if autoregressive and self._config.model == "svg":
            sampled_losses.sort(key=lambda x: x["autoreg_psnr"], reverse=True)
        best_loss = sampled_losses[0]

        for k, v in best_loss.items():
            best_loss[k] /= floor(T / window)
        return best_loss

    @torch.no_grad()
    def _eval_step(self, data, autoregressive=False):
        """
        Evals over a snippet of video of length n_past + n_future
        autoregressive: use model's outputs as input for next timestep
        """
        # one step evaluation loss
        cf = self._config

        x = data["images"]
        states = data["states"]
        ac = data["actions"]
        true_masks = data["masks"]
        masks = data["pred_masks"]
        if cf.model_use_heatmap:
            heatmaps = data["heatmaps"]
        robot_name = data["robot"]
        bs = min(cf.test_batch_size, x.shape[1])
        # initialize the recurrent states
        self.model.init_hidden(bs)

        losses = defaultdict(float)
        k_losses = defaultdict(float)
        robot_name = np.array(robot_name)
        all_robots = set(robot_name)
        x_pred = skip = None
        prefix = "autoreg" if autoregressive else "1step"

        for i in range(1, cf.n_eval):
            if autoregressive and i > 1:
                x_j = x_pred.clone()
            else:
                x_j = x[i - 1]
            # let j be i - 1, or previous timestep
            m_j, r_j, a_j = masks[i - 1], states[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], masks[i], states[i]
            hm_j, hm_i = None, None
            if cf.model_use_heatmap:
                hm_j, hm_i = heatmaps[i - 1], heatmaps[i]

            if cf.model == "copy":
                x_pred = self.model(x_j, m_j, x_i, m_i)
            else:
                # zero out robot pixels in input for norobot cost
                x_j_black, x_i_black = x_j, x_i
                if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                    x_j_black = zero_robot_region(m_j, x_j)
                    x_i_black = zero_robot_region(m_i, x_i)

                if cf.last_frame_skip:
                    skip = None

                m_in = m_j
                if cf.model_use_future_mask:
                    m_in = torch.cat([m_j, m_i], 1)
                r_in = r_j
                if cf.model_use_future_robot_state:
                    r_in = (r_j, r_i)
                hm_in = hm_j
                if cf.model_use_future_heatmap:
                    hm_in = torch.cat([hm_j, hm_i], 1)

                if cf.model == "det":
                    x_pred, curr_skip = self.model(x_j_black, m_in, r_j, a_j, skip)
                elif cf.model == "svg":
                    m_next_in = m_i
                    if cf.model_use_future_mask:
                        m_next_in = m_i.repeat(1, 2, 1, 1)
                    hm_next_in = hm_i
                    if cf.model_use_future_heatmap:
                        hm_next_in = hm_i.repeat(1, 2, 1, 1)
                    out = self.model(
                        x_j_black,
                        m_in,
                        r_in,
                        hm_in,
                        a_j,
                        x_i_black,
                        m_next_in,
                        r_i,
                        hm_next_in,
                        skip,
                        force_use_prior=True,
                    )
                    x_pred, curr_skip, mu, logvar, mu_p, logvar_p = out

                x_pred, x_pred_mask = x_pred[:, :3], x_pred[:, 3].unsqueeze(1)
                x_pred = (1 - x_pred_mask) * x_j + (x_pred_mask) * x_pred
                if i <= cf.n_past:
                    # overwrite skip with most recent conditioning frame skip
                    skip = curr_skip
            if cf.multiview:
                num_views = x_pred.shape[2] // cf.image_width
                for n in range(num_views):
                    start, end = n * cf.image_width, (n + 1) * cf.image_width
                    view_pred = x_pred[:, :, start:end, :]
                    view = x[i][:, :, start:end, :]
                    view_mask = masks[i][:, :, start:end, :]
                    view_loss = self._recon_loss(view_pred, view, view_mask)
                    view_loss_scalar = view_loss.cpu().item()
                    robot_mse = robot_mse_criterion(view_pred, view, view_mask)
                    robot_mse_scalar = robot_mse.cpu().item()
                    world_mse = world_mse_criterion(view_pred, view, view_mask)
                    world_mse_scalar = world_mse.cpu().item()
                    losses[f"{prefix}_view_{n}_robot"] += robot_mse_scalar
                    losses[f"{prefix}_view_{n}_world"] += world_mse_scalar
                    losses[f"{prefix}_view_{n}_recon"] += view_loss_scalar
                    losses[f"{prefix}_total_recon_loss"] += view_loss_scalar
                    losses[f"{prefix}_total_robot_loss"] += robot_mse_scalar
                    losses[f"{prefix}_total_world_loss"] += world_mse_scalar
            else:
                view_loss = self._recon_loss(x_pred, x[i], true_masks[i])
                losses[f"{prefix}_recon_loss"] += view_loss.cpu().item()
                robot_mse = robot_mse_criterion(x_pred, x[i], true_masks[i])
                world_mse = world_mse_criterion(x_pred, x[i], true_masks[i])
                losses[f"{prefix}_robot_loss"] += robot_mse.cpu().item()
                world_mse_value = world_mse.cpu().item()
                losses[f"{prefix}_world_loss"] += world_mse_value

                # black out robot with true mask before computing psnr, ssim
                x_pred_black = zero_robot_region(true_masks[i], x_pred)
                x_i_black = zero_robot_region(true_masks[i], x_i)

                batch_p = psnr(x_i_black.clamp(0, 1), x_pred_black.clamp(0, 1))
                # batch_p = world_psnr_criterion(x_pred, x[i], true_masks[i])
                # for file, p in zip(data["file_path"], batch_p.cpu()):
                    # self.batch_p[file] += p.item()

                p = batch_p.mean().item()
                s = ssim(x_i_black, x_pred_black).mean().item()
                losses[f"{prefix}_psnr"] += p
                losses[f"{prefix}_ssim"] += s

                # k-step rollouts.
                if autoregressive:
                    k_losses[f"{i}_step_psnr"] = p
                    k_losses[f"{i}_step_ssim"] = s
                    k_losses[f"{i}_step_world_loss"] = world_mse_value
                    # for k in range(i, cf.n_eval - 1):
                    #     k_losses[f"{k}_step_psnr"] += p
                    #     k_losses[f"{k}_step_ssim"] += s
                    #     k_losses[f"{k}_step_world_loss"] += world_mse_value

                # robot specific metrics
                for r in all_robots:
                    if len(all_robots) == 1:
                        break
                    r_idx = r == robot_name
                    r_pred = x_pred[r_idx]
                    r_img = x[i][r_idx]
                    r_mask = true_masks[i][r_idx]
                    r_robot_mse = robot_mse_criterion(r_pred, r_img, r_mask)
                    r_world_mse = world_mse_criterion(r_pred, r_img, r_mask)
                    losses[f"{prefix}_{r}_robot_loss"] += r_robot_mse.cpu().item()
                    losses[f"{prefix}_{r}_world_loss"] += r_world_mse.cpu().item()

            if cf.model == "svg":
                kl = kl_criterion(mu, logvar, mu_p, logvar_p, bs)
                losses[f"{prefix}_kld"] += kl.cpu().item()

        for k, v in losses.items():
            losses[k] = v / (cf.n_eval - 1)  # don't count the first step

        # temp_k_losses = {}
        # for k, v in k_losses.items():
        #     num_steps = float(k[0])
        #     temp_k_losses[k] = v / num_steps
        losses.update(k_losses)
        return losses

    def train(self):
        """Training, Evaluation, Checkpointing loop"""
        cf = self._config
        if cf.model == "copy":
            self.train_copy_baseline()
            return

        # load models and dataset
        self._step = self._load_checkpoint(cf.dynamics_model_ckpt)
        self._setup_data()
        T = 15
        window = cf.n_past + cf.n_future
        total = cf.niter * cf.epoch_size * floor(T / window)
        desc = "batches seen"
        self.progress = tqdm(initial=self._step, total=total, desc=desc)

        # start training
        for epoch in range(cf.niter):
            self.model.train()
            # number of batches in 1 epoch
            for i in range(cf.epoch_size):
                data = next(self.training_batch_generator)
                info = self._train_video(data)
                if self._scheduled_sampling:
                    info["sample_schedule"] = self._schedule_prob()[0]
                self._step += self.steps_per_train_video

                if i == cf.epoch_size - 1:
                    self.plot(data, epoch, "train")
                    # self.plot_rec(data, epoch, "train")

                wandb.log({f"train/{k}": v for k, v in info.items()}, step=self._step)
                self.progress.update(self.steps_per_train_video)

            if epoch % cf.checkpoint_interval == 0 and epoch > 0:
                self._logger.info(f"Saving checkpoint {epoch}")
                self._save_checkpoint()

            # always eval at first epoch to get early eval result
            if epoch % cf.eval_interval == 0:
                # plot and evaluate on test set
                start = time()
                self.model.eval()
                info = self._compute_epoch_metrics(self.test_loader, "test")
                end = time()
                data_time = end - start
                print(f"eval time {data_time:.2f}")
                wandb.log(info, step=self._step)
                test_data = next(self.testing_batch_generator)
                self.plot(test_data, epoch, "test")
                if cf.experiment in ["train_sawyer_multiview", "train_robonet"]:
                    info = self._compute_epoch_metrics(self.transfer_loader, "transfer")
                    wandb.log(info, step=self._step)
                    transfer_data = next(self.transfer_batch_generator)
                    self.plot(transfer_data, epoch, "transfer")
        self._logger.info(f"Saving last checkpoint {epoch}")
        self._save_checkpoint()

    def train_copy_baseline(self):
        """Compute metrics for copy baseline"""
        cf = self._config
        # load models and dataset
        self._step = self._load_checkpoint(cf.dynamics_model_ckpt)
        self._setup_data()
        epoch = 0
        # train set
        self.model.train()
        train_info = self._compute_epoch_metrics(self.train_loader, "train")
        train_data = next(self.training_batch_generator)
        self.plot(train_data, epoch, "train")
        # self.plot_rec(train_data, epoch, "train")
        # test set
        self.model.eval()
        test_info = self._compute_epoch_metrics(self.test_loader, "test")
        test_data = next(self.testing_batch_generator)
        self.plot(test_data, epoch, "test")
        # self.plot_rec(test_data, epoch, "test")

        # plot 2 points to make horizontal line
        wandb.log(train_info, step=0)
        wandb.log(test_info, step=0)
        # transfer
        if cf.experiment in ["train_sawyer_multiview"]:
            transfer_info = self._compute_epoch_metrics(
                self.transfer_loader, "transfer"
            )
            transfer_data = next(self.transfer_batch_generator)
            self.plot(transfer_data, epoch, "transfer")
            wandb.log(transfer_info, step=0)
            wandb.log(transfer_info, step=500000)
        wandb.log(train_info, step=500000)
        wandb.log(test_info, step=500000)

    def _save_checkpoint(self):
        path = os.path.join(self._config.log_dir, f"ckpt_{self._step}.pt")
        data = {
            # "background_model": self.background_model.state_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
        }
        torch.save(data, path)

    def _load_robot_model_checkpoint(self, ckpt_path):
        # load given ckpt path
        print(f"Loading Robot Model ckpt {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self._device)
        self.joint_model.load_state_dict(ckpt["joint_model"])
        self.gripper_model.load_state_dict(ckpt["gripper_model"])

    def _load_checkpoint(self, ckpt_path=None):
        """
        Either load a given checkpoint path, or find the most recent checkpoint file
        in the log dir and start from there

        Returns the training step
        """

        def get_recent_ckpt_path(base_dir):
            from glob import glob

            files = glob(os.path.join(base_dir, "*.pt"))
            files.sort()
            if len(files) == 0:
                return None, None
            # assume filename is ckpt_X.pt
            max_step = 0
            path = None
            for f in files:
                name = f.split(".")[0]
                num = int(name.rsplit("_", 1)[-1])
                if num > max_step:
                    max_step = num
                    path = f
            return path, max_step

        if ckpt_path is None:
            # check for most recent ckpt in folder
            ckpt_path, ckpt_num = get_recent_ckpt_path(self._config.log_dir)
            if ckpt_path is None:
                print("Randomly initializing Model")
                return 0
            else:
                print(f"Loading most recent ckpt: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=self._device)
                # self.background_model.load_state_dict(ckpt["background_model"])
                self.model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                step = ckpt["step"]
                return step
        else:
            # load given ckpt path
            print(f"Loading ckpt {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self._device)
            # self.background_model.load_state_dict(ckpt["background_model"])
            self.model.load_state_dict(ckpt["model"])
            if "finetune" in self._config.experiment:
                step = 0
            else:
                step = ckpt["step"]
                self.optimizer.load_state_dict(ckpt["optimizer"])
            return step

    def _setup_data(self):
        """
        Setup the dataset and dataloaders
        """
        if self._config.experiment == "train_robonet":
            from src.dataset.locobot.locobot_singleview_dataloader import (
                create_transfer_loader,
            )
            from src.dataset.robonet.robonet_dataloaders import create_loaders

            # measure zero shot performance on unseen locobot
            self.transfer_loader = create_transfer_loader(self._config)
            self.transfer_batch_generator = get_batch(
                self.transfer_loader, self._device
            )

        elif self._config.experiment == "train_sawyer_multiview":
            from src.dataset.sawyer.sawyer_dataloaders import (
                create_loaders,
                create_transfer_loader,
            )

            # measure zero shot performance on unseen sawyer viewpoint
            self.transfer_loader = create_transfer_loader(self._config)
            self.transfer_batch_generator = get_batch(
                self.transfer_loader, self._device
            )
        elif self._config.experiment == "finetune_sawyer_view":
            from src.dataset.sawyer.sawyer_dataloaders import (
                create_finetune_loaders as create_loaders,
            )
        elif self._config.experiment == "finetune_widowx":
            from src.dataset.widowx.widowx_dataloaders import (
                create_finetune_loaders as create_loaders,
            )
        elif self._config.experiment == "train_locobot_singleview":
            from src.dataset.locobot.locobot_singleview_dataloader import create_loaders
        elif self._config.experiment == "finetune_locobot":
            from src.dataset.locobot.locobot_singleview_dataloader import create_finetune_loaders as create_loaders
        elif self._config.experiment == "train_locobot_table":
            from src.dataset.locobot.locobot_table_dataloaders import create_loaders
        elif self._config.experiment == "train_locobot_pick":
            from src.dataset.locobot.locobot_pick_dataloaders import create_loaders
        else:
            raise NotImplementedError(self._config.experiment)
        self.train_loader, self.test_loader = create_loaders(self._config)
        # for infinite batching
        self.training_batch_generator = get_batch(self.train_loader, self._device)
        self.testing_batch_generator = get_batch(self.test_loader, self._device)

    @torch.no_grad()
    def plot(self, data, epoch, name, random_start=True, instance=None):
        """Plot the generation with learned prior. Autoregressive output.
        Args:
            data (DataLoader): dictionary from dataloader
            epoch (int): epoch number
            name (str): name of the dataset
            random_start (bool, optional): Choose a random timestep as the starting frame
            instance: idx when more than one gif is generated per epoch
        """
        cf = self._config
        x = data["images"]
        states = data["states"]
        ac = data["actions"]
        mask = data["masks"]
        qpos = data["qpos"]
        robot = data["robot"]
        if cf.model_use_heatmap:
            heatmaps = data["heatmaps"]
        nsample = 1
        if cf.model == "svg":
            nsample = 3

        b = min(x.shape[1], 25)
        # first frame of all videos
        start = 0
        video_len = cf.n_eval
        if name in ["comparison", "train"]:
            video_len = cf.n_past + cf.n_future
        end = start + video_len
        if random_start:
            offset = x.shape[0] - video_len
            start = self._plot_rng.randint(0, offset + 1, size=b)
            end = start + video_len
        # truncate batch by time and batch dim
        x = torch.stack([x[s:e, i] for i, (s, e) in enumerate(zip(start, end))], 1)
        states = torch.stack([states[s:e, i] for i, (s, e) in enumerate(zip(start, end))], 1)
        ac = torch.stack([ac[s:e-1, i] for i, (s, e) in enumerate(zip(start, end))], 1)
        mask = torch.stack([mask[s:e, i] for i, (s, e) in enumerate(zip(start, end))], 1)
        qpos = torch.stack([qpos[s:e, i] for i, (s, e) in enumerate(zip(start, end))], 1)
        folder = data["folder"][:b]
        robot = robot[:b]
        if cf.model_use_heatmap:
            heatmaps = torch.stack([heatmaps[s:e, i] for i, (s, e) in enumerate(zip(start, end))], 1)

        if "finetune" in cf.experiment and (cf.model_use_mask or cf.model_use_robot_state):
            input_data = dict(
                states=states,
                actions=ac,
                masks=mask,
                qpos=qpos,
                folder=folder,
                robot=robot,
            )
            input_data["low"] = data["low"][:b]
            input_data["high"] = data["high"][:b]
            if cf.preprocess_action != "raw":
                input_data["raw_low"] = data["raw_low"][:b]
                input_data["raw_high"] = data["raw_high"][:b]
                input_data["raw_actions"] = torch.stack([data["raw_actions"][s:e-1, i] for i, (s, e) in enumerate(zip(start, end))], 1)
                input_data["raw_states"] = torch.stack([data["raw_states"][s:e, i] for i, (s, e) in enumerate(zip(start, end))], 1)

            if cf.experiment == "finetune_locobot":
                out = self.robot_model.predict_batch(input_data)
            elif cf.learned_robot_model:
                out = self._generate_learned_robot_states(input_data)

            if cf.model_use_heatmap:
                states, mask, heatmaps = out
            else:
                states, mask = out

        gen_seq = [[] for i in range(nsample)]
        gen_mask = [[] for i in range(nsample)]
        gt_seq = [x[i] for i in range(len(x))]
        gt_mask = [mask[i] for i in range(len(mask))]

        for s in range(nsample):
            skip = None
            self.model.init_hidden(b)
            if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                x_j = zero_robot_region(mask[0], x[0])
            else:
                x_j = x[0]
            gen_seq[s].append(x_j)
            gen_mask[s].append(mask[0])
            for i in range(1, video_len):
                # let j be i - 1, or previous timestep
                m_j, r_j, a_j = mask[i - 1], states[i - 1], ac[i - 1]
                x_i, m_i, r_i = x[i], mask[i], states[i]
                hm_j, hm_i = None, None
                if cf.model_use_heatmap:
                    hm_j, hm_i = heatmaps[i - 1], heatmaps[i]
                if cf.model == "copy":
                    x_pred = self.model(x_j, m_j, x_i, m_i)
                else:
                    x_j_black, x_i_black = x_j, x_i
                    # zero out robot pixels in input for norobot cost
                    if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                        x_j_black = zero_robot_region(m_j, x_j)
                        x_i_black = zero_robot_region(m_i, x_i)
                    m_in = m_j
                    if cf.model_use_future_mask:
                        m_in = torch.cat([m_j, m_i], 1)
                    r_in = r_j
                    if cf.model_use_future_robot_state:
                        r_in = (r_j, r_i)
                    hm_in = hm_j
                    if cf.model_use_future_heatmap:
                        hm_in = torch.cat([hm_j, hm_i], 1)

                    if cf.last_frame_skip:
                        # overwrite conditioning frame skip if necessary
                        skip = None

                    if cf.model == "det":
                        x_pred, curr_skip = self.model(x_j, m_in, r_j, a_j, skip)
                    elif cf.model == "svg":
                        # don't use posterior.
                        x_i_black, m_next_in, r_i, hm_next_in = None, None, None, None
                        out = self.model(
                            x_j_black,
                            m_in,
                            r_in,
                            hm_in,
                            a_j,
                            x_i_black,
                            m_next_in,
                            r_i,
                            hm_next_in,
                            skip,
                        )
                        x_pred, curr_skip, _, _, _, _ = out

                    x_pred, x_pred_mask = x_pred[:, :3], x_pred[:, 3].unsqueeze(1)
                    x_pred = (1 - x_pred_mask) * x_j + (x_pred_mask) * x_pred

                    if i <= cf.n_past:
                        # store the most recent conditioning frame's skip
                        skip = curr_skip

                if i < cf.n_past:
                    x_j = x_i
                else:
                    x_j = x_pred
                if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                    gen_seq[s].append(zero_robot_region(m_i, x_pred))
                else:
                    gen_seq[s].append(x_j)
                gen_mask[s].append(x_pred_mask)

        to_plot = []
        mask_to_plot = []
        gifs = [[] for t in range(video_len)]
        mask_gifs = [[] for t in range(video_len)]
        nrow = b
        for i in range(nrow):
            # ground truth sequence
            row = []
            mask_row = []
            for t in range(video_len):
                row.append(gt_seq[t][i])
                mask_row.append(gt_mask[t][i])
            to_plot.append(row)
            mask_to_plot.append(mask_row)
            if cf.model == "svg":
                s_list = range(nsample)
            else:
                s_list = [0]
            for t in range(video_len):
                row = []
                mask_row = []
                row.append(gt_seq[t][i])
                mask_row.append(gt_mask[t][i])
                for ss in range(len(s_list)):
                    s = s_list[ss]
                    row.append(gen_seq[s][t][i])
                    mask_row.append(gen_mask[s][t][i])
                gifs[t].append(row)
                mask_gifs[t].append(mask_row)
        # gifs is T x B x S x |I|
        if instance is None:
            fname = os.path.join(cf.plot_dir, f"{name}_{epoch}.gif")
            mask_fname = os.path.join(cf.plot_dir, f"{name}_{epoch}_masks.gif")
        else:
            fname = os.path.join(cf.plot_dir, f"{name}_ep{epoch}_{instance}.gif")
            mask_fname = os.path.join(cf.plot_dir, f"{name}_ep{epoch}_{instance}_masks.gif")
        save_gif(fname, gifs)
        # batch_p = []
        # for k,v in self.batch_p.items():
        #     batch_p.append(f"{(v/5):.2f}")
        # text = [batch_p for _ in range(len(gifs))]
        # save_gif_with_text(fname, gifs, text)
        # save_gif(mask_fname, mask_gifs)
        if cf.wandb:
            wandb.log({f"{name}/gifs": wandb.Video(fname, format="gif")}, step=self._step)
            # wandb.log(
            # {f"{name}/masks_gifs": wandb.Video(mask_fname, format="gif")}, step=self._step
            # )

    def predict_video(self, data):
        """Generate predictions for the video
        Used to compute FID scores
        data: video data from dataloader
        Outputs the ground truth and predicted videos
        """
        cf = self._config
        num_samples = 1
        if cf.model == "svg" and "finetune" in cf.experiment:
            num_samples = 3
        x = data["images"]
        T = len(x)
        window = cf.n_eval
        sampled_losses = [
            defaultdict(float) for _ in range(num_samples)
        ]  # list of video sample losses
        for i in range(floor(T / window)):
            s = i * window
            e = (i + 1) * window
            batch_data = {
                "images": x[s:e],
                "states": data["states"][s:e],
                "actions": data["actions"][s : e - 1],
                "masks": data["masks"][s:e],
                "robot": data["robot"],
                "qpos": data["qpos"][s:e],
                "folder": data["folder"],
            }
            if cf.model_use_heatmap:
                batch_data["heatmaps"] = data["heatmaps"][s:e]

            if "finetune" in cf.experiment:
                if cf.preprocess_action != "raw":
                    batch_data["raw_actions"] = data["raw_actions"][s : e - 1]
                batch_data["low"] = data["low"]
                batch_data["high"] = data["high"]

                if cf.experiment == "finetune_locobot":
                    # use analytical model
                    # inputs: current qpos, state, list of actions
                    # outputs: list of states, list of masks
                    out = self.robot_model.predict_batch(batch_data)
                elif cf.learned_robot_model:
                    out = self._generate_learned_robot_states(batch_data)

                if cf.model_use_heatmap:
                    states, masks, heatmaps = out
                    batch_data["heatmaps"] = heatmaps
                else:
                    states, masks = out

                batch_data["states"] = states
                # eval also needs true masks for computing world errors
                batch_data["pred_masks"] = masks
            else:  # use true masks for eval rollout
                batch_data["pred_masks"] = batch_data["masks"]

            for sample in range(num_samples):
                losses = self._predict_video(batch_data)
                for k, v in losses.items():
                    if k in ["true_imgs", "gen_imgs"]:
                        if k not in sampled_losses[sample]:
                            sampled_losses[sample][k] = []
                        sampled_losses[sample][k].append(v)
                    else:
                        sampled_losses[sample][k] += v
        # now pick the best sample by world error, and average over frames
        if cf.model == "svg":
            sampled_losses.sort(key=lambda x: x["autoreg_world_loss"])
        best_loss = sampled_losses[0]

        for k, v in best_loss.items():
            if k in ["true_imgs", "gen_imgs"]:
                continue
            best_loss[k] /= floor(T / window)
        return best_loss

    @torch.no_grad()
    def _predict_video(self, data, autoregressive=True):
        """
        Evals over a snippet of video of length n_past + n_future
        autoregressive: use model's outputs as input for next timestep
        """
        # one step evaluation loss
        cf = self._config

        x = data["images"]
        states = data["states"]
        ac = data["actions"]
        true_mask = data["masks"]
        mask = data["pred_masks"]
        if cf.model_use_heatmap:
            heatmaps = data["heatmaps"]
        robot_name = data["robot"]
        bs = min(cf.test_batch_size, x.shape[1])
        # initialize the recurrent states
        self.model.init_hidden(bs)

        losses = defaultdict(float)
        k_losses = defaultdict(float)
        robot_name = np.array(robot_name)
        all_robots = set(robot_name)

        # store for computing FID metric
        all_x_pred_black = []
        all_x_i_black = []

        x_pred = skip = None
        prefix = "autoreg" if autoregressive else "1step"

        for i in range(1, cf.n_eval):
            if autoregressive and i > 1:
                x_j = x_pred.clone()
            else:
                x_j = x[i - 1]
            # let j be i - 1, or previous timestep
            m_j, r_j, a_j = mask[i - 1], states[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], mask[i], states[i]
            hm_j, hm_i = None, None
            if cf.model_use_heatmap:
                hm_j, hm_i = heatmaps[i - 1], heatmaps[i]

            if cf.model == "copy":
                x_pred = self.model(x_j, m_j, x_i, m_i)
            else:
                # zero out robot pixels in input for norobot cost
                x_j_black, x_i_black = x_j, x_i
                if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                    x_j_black = zero_robot_region(m_j, x_j)
                    x_i_black = zero_robot_region(m_i, x_i)

                if cf.last_frame_skip:
                    skip = None

                m_in = m_j
                if cf.model_use_future_mask:
                    m_in = torch.cat([m_j, m_i], 1)
                r_in = r_j
                if cf.model_use_future_robot_state:
                    r_in = (r_j, r_i)
                hm_in = hm_j
                if cf.model_use_future_heatmap:
                    hm_in = torch.cat([hm_j, hm_i], 1)

                if cf.model == "det":
                    x_pred, curr_skip = self.model(x_j_black, m_in, r_j, a_j, skip)
                elif cf.model == "svg":
                    m_next_in = m_i
                    if cf.model_use_future_mask:
                        m_next_in = m_i.repeat(1, 2, 1, 1)
                    hm_next_in = hm_i
                    if cf.model_use_future_heatmap:
                        hm_next_in = hm_i.repeat(1, 2, 1, 1)
                    out = self.model(
                        x_j_black,
                        m_in,
                        r_in,
                        hm_in,
                        a_j,
                        x_i_black,
                        m_next_in,
                        r_i,
                        hm_next_in,
                        skip,
                        force_use_prior=True,
                    )
                    x_pred, curr_skip, mu, logvar, mu_p, logvar_p = out

                x_pred, x_pred_mask = x_pred[:, :3], x_pred[:, 3].unsqueeze(1)
                x_pred = (1 - x_pred_mask) * x_j + (x_pred_mask) * x_pred
                if i <= cf.n_past:
                    # overwrite skip with most recent conditioning frame skip
                    skip = curr_skip
            if cf.multiview:
                num_views = x_pred.shape[2] // cf.image_width
                for n in range(num_views):
                    start, end = n * cf.image_width, (n + 1) * cf.image_width
                    view_pred = x_pred[:, :, start:end, :]
                    view = x[i][:, :, start:end, :]
                    view_mask = mask[i][:, :, start:end, :]
                    view_loss = self._recon_loss(view_pred, view, view_mask)
                    view_loss_scalar = view_loss.cpu().item()
                    robot_mse = robot_mse_criterion(view_pred, view, view_mask)
                    robot_mse_scalar = robot_mse.cpu().item()
                    world_mse = world_mse_criterion(view_pred, view, view_mask)
                    world_mse_scalar = world_mse.cpu().item()
                    losses[f"{prefix}_view_{n}_robot"] += robot_mse_scalar
                    losses[f"{prefix}_view_{n}_world"] += world_mse_scalar
                    losses[f"{prefix}_view_{n}_recon"] += view_loss_scalar
                    losses[f"{prefix}_total_recon_loss"] += view_loss_scalar
                    losses[f"{prefix}_total_robot_loss"] += robot_mse_scalar
                    losses[f"{prefix}_total_world_loss"] += world_mse_scalar
            else:
                view_loss = self._recon_loss(x_pred, x[i], true_mask[i])
                losses[f"{prefix}_recon_loss"] += view_loss.cpu().item()
                robot_mse = robot_mse_criterion(x_pred, x[i], true_mask[i])
                world_mse = world_mse_criterion(x_pred, x[i], true_mask[i])
                losses[f"{prefix}_robot_loss"] += robot_mse.cpu().item()
                world_mse_value = world_mse.cpu().item()
                losses[f"{prefix}_world_loss"] += world_mse_value

                # black out robot with true mask before computing psnr, ssim
                x_pred_black = zero_robot_region(true_mask[i], x_pred)
                x_i_black = zero_robot_region(true_mask[i], x_i)

                all_x_pred_black.append(x_pred_black)
                all_x_i_black.append(x_i_black)

                p = psnr(x_i_black.clamp(0, 1), x_pred_black.clamp(0, 1)).mean().item()
                s = ssim(x_i_black, x_pred_black).mean().item()
                losses[f"{prefix}_psnr"] += p
                losses[f"{prefix}_ssim"] += s

                # k-step rollouts.
                if autoregressive:
                    for k in range(i, cf.n_eval - 1):
                        k_losses[f"{k}_step_psnr"] += p
                        k_losses[f"{k}_step_ssim"] += s
                        k_losses[f"{k}_step_world_loss"] += world_mse_value

                # robot specific metrics
                for r in all_robots:
                    if len(all_robots) == 1:
                        break
                    r_idx = r == robot_name
                    r_pred = x_pred[r_idx]
                    r_img = x[i][r_idx]
                    r_mask = true_mask[i][r_idx]
                    r_robot_mse = robot_mse_criterion(r_pred, r_img, r_mask)
                    r_world_mse = world_mse_criterion(r_pred, r_img, r_mask)
                    losses[f"{prefix}_{r}_robot_loss"] += r_robot_mse.cpu().item()
                    losses[f"{prefix}_{r}_world_loss"] += r_world_mse.cpu().item()

            if cf.model == "svg":
                kl = kl_criterion(mu, logvar, mu_p, logvar_p, bs)
                losses[f"{prefix}_kld"] += kl.cpu().item()

        for k, v in losses.items():
            losses[k] = v / (cf.n_eval - 1)  # don't count the first step

        temp_k_losses = {}
        for k, v in k_losses.items():
            num_steps = float(k[0])
            temp_k_losses[k] = v / num_steps
        losses.update(temp_k_losses)
        # make this B x T x C x H x W
        all_x_pred_black = torch.stack(all_x_pred_black).transpose(0, 1)
        all_x_i_black = torch.stack(all_x_i_black).transpose(0, 1)

        losses["gen_imgs"] = (
            (255 * all_x_pred_black)
            .permute(0, 1, 3, 4, 2)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        losses["true_imgs"] = (
            (255 * all_x_i_black).permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
        )
        return losses


def make_log_folder(config):
    # make folder for exp logs
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red,bold",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    # only logs to console
    logger = colorlog.getLogger("console")
    logger.setLevel(logging.DEBUG)

    ch = colorlog.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    config.log_dir = os.path.join(config.log_dir, config.jobname)
    logger.info(f"Create log directory: {config.log_dir}")
    os.makedirs(config.log_dir, exist_ok=True)

    config.plot_dir = os.path.join(config.log_dir, "plot")
    os.makedirs(config.plot_dir, exist_ok=True)

    config.video_dir = os.path.join(config.log_dir, "video")
    os.makedirs(config.video_dir, exist_ok=True)

    config.trajectory_dir = os.path.join(config.log_dir, "trajectory")
    os.makedirs(config.trajectory_dir, exist_ok=True)

    # create the file / console logger
    filelogger = colorlog.getLogger("file/console")
    filelogger.setLevel(logging.DEBUG)
    logfile_path = os.path.join(config.log_dir, "log.txt")
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s @l%(lineno)d: %(message)s", "%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)

    filelogger.addHandler(fh)
    filelogger.addHandler(ch)


if __name__ == "__main__":
    # import torch.multiprocessing as mp
    from src.config import argparser

    config, _ = argparser()
    make_log_folder(config)
    trainer = PredictionTrainer(config)
    trainer.train()
