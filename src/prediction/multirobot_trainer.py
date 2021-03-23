import logging
import os
from collections import defaultdict
from functools import partial

import torchvision.transforms as tf
from src.dataset.multirobot_dataset import get_batch, process_batch
from src.env.robotics.masks.baxter_mask_env import BaxterMaskEnv
from src.env.robotics.masks.sawyer_mask_env import SawyerMaskEnv
from src.env.robotics.masks.widowx_mask_env import WidowXMaskEnv
from src.prediction.models.dynamics import (
    CopyModel,
    DeterministicConvModel,
    GripperStatePredictor,
    JointPosPredictor,
    SVGConvModel,
)
from src.utils.camera_calibration import world_to_camera_dict

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
)
from src.utils.metrics import psnr, ssim
from src.utils.plot import save_gif, save_tensors_image
from torch import optim
from tqdm import tqdm


class MultiRobotPredictionTrainer(object):
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
            self.model = SVGConvModel(cf).to(self._device)
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

        if cf.learned_robot_model:
            # learned robot models for evaluation
            self.joint_model = JointPosPredictor(cf).to(self._device)
            self.gripper_model = GripperStatePredictor(cf).to(self._device)
            self._load_robot_model_checkpoint(cf.robot_model_ckpt)

    def _schedule_prob(self):
        """Returns probability of using ground truth"""
        # assume 50k max training steps
        # https://www.desmos.com/calculator/bo4aoyqje1
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

    def _recon_loss(self, prediction, target, mask=None):
        if self._config.reconstruction_loss == "mse":
            return mse_criterion(prediction, target)
        elif self._config.reconstruction_loss == "l1":
            return l1_criterion(prediction, target)
        elif self._config.reconstruction_loss == "dontcare_mse":
            robot_weight = self._config.robot_pixel_weight
            return dontcare_mse_criterion(prediction, target, mask, robot_weight)
        elif self._config.reconstruction_loss == "dontcare_l1":
            robot_weight = self._config.robot_pixel_weight
            return dontcare_l1_criterion(prediction, target, mask, robot_weight)
        else:
            raise NotImplementedError(f"{self._config.reconstruction_loss}")

    def _zero_robot_region(self, mask, image, inplace=True):
        """
        Set the robot region to zero
        """
        robot_mask = mask.type(torch.bool)
        robot_mask = robot_mask.repeat(1, 3, 1, 1)
        if not inplace:
            image = image.clone()
        image[robot_mask] *= 0
        return image

    @torch.no_grad()
    def _generate_learned_masks_states(self, data):
        """
        Use the learned robot model to generate masks and states for
        training / eval.
        """
        cf = self._config
        states = data["states"]
        ac = data["actions"]
        mask = data["masks"]
        qpos = data["qpos"]
        viewpoints = set(data["file_name"])
        if not hasattr(self, "renderers"):
            self.renderers = {}
        for v in viewpoints:
            if v in self.renderers:
                continue
            if cf.training_regime == "singlerobot":
                env = SawyerMaskEnv()
                cam_ext = world_to_camera_dict[f"sawyer_{v}"]
            elif cf.training_regime == "finetune":
                env = BaxterMaskEnv()
                env.arm = "left"
                cam_ext = world_to_camera_dict[f"baxter_left"]
            elif cf.training_regime == "finetune_widowx":
                env = WidowXMaskEnv()
                cam_ext = world_to_camera_dict[f"widowx1"]
            elif cf.training_regime == "finetune_sawyer_view":
                env = SawyerMaskEnv()
                cam_ext = world_to_camera_dict[f"sawyer_{v}"]
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
                vp = data["file_name"][b]
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

        # generate learned masks for finetuning
        # if cf.learned_robot_model and "finetune" in cf.training_regime:
        #     if not hasattr(self, "robot_state_mask_cache"):
        #         self._state_mask_cache = {}

        #     # collect the indices of files that are not cached,
        #     # group them into a batch, and then generate them
        #     not_cached_idxs = []
        #     for idx, file_name in enumerate(data["file_name"]):
        #         if file_name in self._state_mask_cache:
        #             states, masks = self._state_mask_cache[file_name]
        #             data["states"][:, idx] = states
        #             data["masks"][:, idx] = masks
        #         else:
        #             not_cached_idxs.append(idx)
        #     # collect the data dict of not cached files for generation
        #     not_cached_data = {}
        #     not_cached_data["states"] = torch.stack([data["states"][:, i] for i in not_cached_idxs], 1)
        #     not_cached_data["actions"] = torch.stack([data["actions"][:, i] for i in not_cached_idxs], 1)
        #     not_cached_data["masks"] = torch.stack([data["masks"][:, i] for i in not_cached_idxs], 1)
        #     not_cached_data["qpos"] = torch.stack([data["qpos"][:, i] for i in not_cached_idxs], 1)
        #     not_cached_data["file_name"] =[data["file_name"][i] for i in not_cached_idxs]

        #     states, masks = self._generate_learned_masks_states(not_cached_data)
        #     for idx in not_cached_idxs:
        #         s = data["states"][:, idx] = states[:, idx]
        #         m = data["masks"][:, idx] = masks[:, idx]
        #         file_name = not_cached_data["file_name"][idx]
        #         self._state_mask_cache["file_name"] = (s,m)

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
                "file_name": data["file_name"],
            }
            if cf.learned_robot_model and "finetune" in cf.training_regime:
                states, masks = self._generate_learned_masks_states(batch_data)
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
        robot_name = data["robot"]
        robot_name = np.array(robot_name)
        all_robots = set(robot_name)
        x_pred = None
        skip = None

        self.model.zero_grad()
        bs = min(cf.batch_size, x.shape[1])
        self.model.init_hidden(bs)  # initialize the recurrent states

        # background mask
        if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
            self._zero_robot_region(mask[0], x[0])
        for i in range(1, cf.n_past + cf.n_future):
            if i > 1:
                x_j = x[i - 1] if self._use_true_token() else x_pred.clone().detach()
            else:
                x_j = x[i - 1]
            # let j be i - 1, or previous timestep
            m_j, r_j, a_j = mask[i - 1], states[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], mask[i], states[i]

            # zero out robot pixels in input for norobot cost
            x_j_black, x_i_black = x_j, x_i
            if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                x_j_black = self._zero_robot_region(m_j, x_j, False)
                x_i_black = self._zero_robot_region(m_i, x_i, False)

            if cf.last_frame_skip:
                skip = None

            m_in = m_j
            if cf.model_use_future_mask:
                m_in = torch.cat([m_j, m_i], 1)
            r_in = r_j
            if cf.model_use_future_robot_state:
                r_in = (r_j, r_i)

            if cf.model == "det":
                x_pred, curr_skip = self.model(x_j_black, m_in, r_j, a_j, skip)
            elif cf.model == "svg":
                m_next_in = m_i
                if cf.model_use_future_mask:
                    m_next_in = m_i.repeat(1, 2, 1, 1)
                out = self.model(
                    x_j_black, m_in, r_in, a_j, x_i_black, m_next_in, r_i, skip
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
            # info = self._eval_video(data)
            # for k, v in info.items():
            #     losses[k].append(v)
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
        num_samples = 1
        if (
            autoregressive
            and self._config.model == "svg"
            and "finetune" in self._config.training_regime
        ):
            num_samples = 3
        x = data["images"]
        T = len(x)
        window = self._config.n_eval
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
                "file_name": data["file_name"],
            }
            for sample in range(num_samples):
                losses = self._eval_step(batch_data, autoregressive)
                for k, v in losses.items():
                    sampled_losses[sample][k] += v

        # now pick the best sample by world error, and average over frames
        if autoregressive and self._config.model == "svg":
            sampled_losses.sort(key=lambda x: x["autoreg_world_loss"])
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
        true_mask = mask = data["masks"]
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
        if autoregressive and cf.learned_robot_model:
            states, mask = self._generate_learned_masks_states(data)

        if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
            self._zero_robot_region(mask[0], x[0])
        for i in range(1, cf.n_eval):
            if autoregressive and i > 1:
                x_j = x_pred.clone()
            else:
                x_j = x[i - 1]
            # let j be i - 1, or previous timestep
            m_j, r_j, a_j = mask[i - 1], states[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], mask[i], states[i]

            if cf.model == "copy":
                x_pred = self.model(x_j, m_j, x_i, m_i)
            else:
                # zero out robot pixels in input for norobot cost
                x_j_black, x_i_black = x_j, x_i
                if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                    x_j_black = self._zero_robot_region(m_j, x_j, False)
                    x_i_black = self._zero_robot_region(m_i, x_i, False)

                if cf.last_frame_skip:
                    skip = None

                m_in = m_j
                if cf.model_use_future_mask:
                    m_in = torch.cat([m_j, m_i], 1)
                r_in = r_j
                if cf.model_use_future_robot_state:
                    r_in = (r_j, r_i)

                if cf.model == "det":
                    x_pred, curr_skip = self.model(x_j_black, m_in, r_j, a_j, skip)
                elif cf.model == "svg":
                    m_next_in = m_i
                    if cf.model_use_future_mask:
                        m_next_in = m_i.repeat(1, 2, 1, 1)
                    out = self.model(
                        x_j_black,
                        m_in,
                        r_in,
                        a_j,
                        x_i_black,
                        m_next_in,
                        r_i,
                        skip,
                        force_use_prior=True
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
                x_pred_black = self._zero_robot_region(true_mask[i], x_pred, False)
                x_i_black = self._zero_robot_region(true_mask[i], x_i, False)

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
        T = 31
        window = cf.n_past + cf.n_future
        total = cf.niter * cf.epoch_size * floor(T / window)
        desc = "batches seen"
        self.progress = tqdm(initial=self._step, total=total, desc=desc)

        # start training
        for epoch in range(cf.niter):
            # self.background_model.train()
            self.model.train()
            # number of batches in 1 epoch
            for i in range(cf.epoch_size):
                # start = time()
                data = next(self.training_batch_generator)
                # end = time()
                # data_time = end - start
                # print("data loading time", data_time)

                # start = time()
                # with torch.autograd.set_detect_anomaly(True):
                info = self._train_video(data)
                # end = time()
                # update_time = end - start
                # print("network update time", update_time)
                # for k, v in info.items():
                #     epoch_losses[f"train/epoch_{k}"] += v
                if self._scheduled_sampling:
                    info["sample_schedule"] = self._schedule_prob()[0]
                self._step += self.steps_per_train_video

                if i == cf.epoch_size - 1:
                    self.plot(data, epoch, "train")
                    # self.plot_rec(data, epoch, "train")

                wandb.log({f"train/{k}": v for k, v in info.items()}, step=self._step)
                self.progress.update(self.steps_per_train_video)

            # log epoch statistics
            # wandb.log(epoch_losses, step=self._step)
            # epoch_log_str = ""
            # for k, v in epoch_losses.items():
            #     epoch_log_str += f"{k}: {v}, "
            # self._logger.info(epoch_log_str)
            # checkpoint
            # self._epoch_save_fid_images(random_snippet=True)
            if epoch % cf.checkpoint_interval == 0 and epoch > 0:
                self._logger.info(f"Saving checkpoint {epoch}")
                self._save_checkpoint()

            # always eval at first epoch to get early eval result
            if epoch % cf.eval_interval == 0:
                # plot and evaluate on test set
                # self.background_model.eval()
                start = time()
                self.model.eval()
                info = self._compute_epoch_metrics(self.test_loader, "test")
                end = time()
                data_time = end - start
                print(f"eval time {data_time:.2f}")
                wandb.log(info, step=self._step)
                test_data = next(self.testing_batch_generator)
                self.plot(test_data, epoch, "test")
                # self.plot_rec(test_data, epoch, "test")
                comp_data = next(self.comp_batch_generator)
                self.plot(comp_data, epoch, "comparison")

                if cf.training_regime in ["singlerobot", "train_sawyer_multiview"]:
                    info = self._compute_epoch_metrics(self.transfer_loader, "transfer")
                    wandb.log(info, step=self._step)
                    transfer_data = next(self.transfer_batch_generator)
                    self.plot(transfer_data, epoch, "transfer")

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
        if cf.training_regime in ["singlerobot", "train_sawyer_multiview"]:
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
            if "finetune" in self._config.training_regime:
                step = 0
            else:
                step = ckpt["step"]
                self.optimizer.load_state_dict(ckpt["optimizer"])
            return step

    def _setup_data(self):
        """
        Setup the dataset and dataloaders
        """
        if self._config.training_regime == "multirobot":
            from src.dataset.multirobot_dataloaders import create_loaders
        elif self._config.training_regime == "singlerobot":
            from src.dataset.finetune_multirobot_dataloaders import create_loaders
            from src.dataset.finetune_widowx_dataloaders import create_transfer_loader

            # measure zero shot performance on transfer data
            self.transfer_loader = create_transfer_loader(self._config)
            self.transfer_batch_generator = get_batch(
                self.transfer_loader, self._device
            )
        elif self._config.training_regime == "finetune":
            from src.dataset.finetune_multirobot_dataloaders import (
                create_finetune_loaders as create_loaders,
            )
        elif self._config.training_regime == "train_sawyer_multiview":
            from src.dataset.sawyer_multiview_dataloaders import (
                create_loaders,
                create_transfer_loader,
            )

            # measure zero shot performance on transfer data
            self.transfer_loader = create_transfer_loader(self._config)
            self.transfer_batch_generator = get_batch(
                self.transfer_loader, self._device
            )
        elif self._config.training_regime == "finetune_sawyer_view":
            from src.dataset.sawyer_multiview_dataloaders import (
                create_finetune_loaders as create_loaders,
            )
        elif self._config.training_regime == "finetune_widowx":
            from src.dataset.finetune_widowx_dataloaders import (
                create_finetune_loaders as create_loaders,
            )
        else:
            raise NotImplementedError(self._config.training_regime)
        self.train_loader, self.test_loader, comp_loader = create_loaders(self._config)
        # for infinite batching
        self.training_batch_generator = get_batch(self.train_loader, self._device)
        self.testing_batch_generator = get_batch(self.test_loader, self._device)
        self.comp_batch_generator = get_batch(comp_loader, self._device)

    @torch.no_grad()
    def plot(self, data, epoch, name, random_start=True):
        """Plot the generation with learned prior. Autoregressive output.
        Args:
            data (DataLoader): dictionary from dataloader
            epoch (int): epoch number
            name (str): name of the dataset
            random_start (bool, optional): Choose a random timestep as the starting frame
        """
        cf = self._config
        x = data["images"]
        states = data["states"]
        ac = data["actions"]
        mask = data["masks"]
        qpos = data["qpos"]
        nsample = 1
        if cf.model == "svg":
            nsample = 3

        b = min(x.shape[1], 10)
        # first frame of all videos
        start = 0
        video_len = cf.n_eval
        if name in ["comparison", "train"]:
            video_len = cf.n_past + cf.n_future
        end = start + video_len
        if random_start:
            offset = x.shape[0] - video_len
            start = self._plot_rng.randint(0, offset + 1)
            end = start + video_len
        # truncate batch by time and batch dim
        x = x[start:end, :b]
        states = states[start:end, :b]
        ac = ac[start : end - 1, :b]
        mask = mask[start:end, :b]
        qpos = qpos[start:end, :b]
        file_name = data["file_name"][:b]

        if cf.learned_robot_model:
            data = dict(
                images=x,
                states=states,
                actions=ac,
                masks=mask,
                qpos=qpos,
                file_name=file_name,
            )
            states, mask = self._generate_learned_masks_states(data)

        gen_seq = [[] for i in range(nsample)]
        gen_mask = [[] for i in range(nsample)]
        gt_seq = [x[i] for i in range(len(x))]
        gt_mask = [mask[i] for i in range(len(mask))]

        for s in range(nsample):
            skip = None
            self.model.init_hidden(b)
            if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                self._zero_robot_region(mask[0], x[0])
            gen_seq[s].append(x[0])
            gen_mask[s].append(mask[0])
            x_j = x[0]
            for i in range(1, video_len):
                # let j be i - 1, or previous timestep
                m_j, r_j, a_j = mask[i - 1], states[i - 1], ac[i - 1]
                x_i, m_i, r_i = x[i], mask[i], states[i]
                if cf.model == "copy":
                    x_pred = self.model(x_j, m_j, x_i, m_i)
                else:
                    # zero out robot pixels in input for norobot cost
                    if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                        self._zero_robot_region(mask[i - 1], x_j)
                        self._zero_robot_region(mask[i], x[i])
                    m_in = m_j
                    if cf.model_use_future_mask:
                        m_in = torch.cat([m_j, m_i], 1)
                    r_in = r_j
                    if cf.model_use_future_robot_state:
                        r_in = (r_j, r_i)

                    if cf.last_frame_skip:
                        # overwrite conditioning frame skip if necessary
                        skip = None

                    if cf.model == "det":
                        x_pred, curr_skip = self.model(x_j, m_in, r_j, a_j, skip)
                    elif cf.model == "svg":
                        # don't use posterior.
                        x_i, m_next_in, r_i = None, None, None
                        out = self.model(x_j, m_in, r_in, a_j, x_i, m_next_in, r_i, skip)
                        x_pred, curr_skip, _, _, _, _ = out

                    x_pred, x_pred_mask = x_pred[:, :3], x_pred[:, 3].unsqueeze(1)
                    x_pred = (1 - x_pred_mask) * x_j + (x_pred_mask) * x_pred

                    if i <= cf.n_past:
                        # store the most recent conditioning frame's skip
                        skip = curr_skip

                    if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                        self._zero_robot_region(mask[i], x_pred)
                if i < cf.n_past:
                    x_j = x_i
                else:
                    x_j = x_pred
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
        fname = os.path.join(cf.plot_dir, f"{name}_{epoch}.gif")
        save_gif(fname, gifs)
        wandb.log({f"{name}/gifs": wandb.Video(fname, format="gif")}, step=self._step)

        fname = os.path.join(cf.plot_dir, f"{name}_{epoch}_masks.gif")
        save_gif(fname, mask_gifs)
        wandb.log(
            {f"{name}/masks_gifs": wandb.Video(fname, format="gif")}, step=self._step
        )

    def predict_video(self, data):
        """Generate predictions for the video
        Used to compute FID scores
        data: video data from dataloader
        Outputs the ground truth and predicted videos
        """
        num_samples = 1
        if (self._config.model == "svg"
            and "finetune" in self._config.training_regime
        ):
            num_samples = 3
        x = data["images"]
        T = len(x)
        window = self._config.n_eval
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
                "file_name": data["file_name"],
            }
            for sample in range(num_samples):
                losses = self._predict_video(batch_data)
                for k, v in losses.items():
                    if k in ["true_imgs", "gen_imgs"]:
                        sampled_losses[sample][k] = v
                    else:
                        sampled_losses[sample][k] += v

        # now pick the best sample by world error, and average over frames
        if self._config.model == "svg":
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
        true_mask = mask = data["masks"]
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
        if autoregressive and cf.learned_robot_model:
            states, mask = self._generate_learned_masks_states(data)

        if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
            self._zero_robot_region(mask[0], x[0])
        for i in range(1, cf.n_eval):
            if autoregressive and i > 1:
                x_j = x_pred.clone()
            else:
                x_j = x[i - 1]
            # let j be i - 1, or previous timestep
            m_j, r_j, a_j = mask[i - 1], states[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], mask[i], states[i]

            if cf.model == "copy":
                x_pred = self.model(x_j, m_j, x_i, m_i)
            else:
                # zero out robot pixels in input for norobot cost
                x_j_black, x_i_black = x_j, x_i
                if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                    x_j_black = self._zero_robot_region(m_j, x_j, False)
                    x_i_black = self._zero_robot_region(m_i, x_i, False)

                if cf.last_frame_skip:
                    skip = None

                m_in = m_j
                if cf.model_use_future_mask:
                    m_in = torch.cat([m_j, m_i], 1)
                r_in = r_j
                if cf.model_use_future_robot_state:
                    r_in = (r_j, r_i)

                if cf.model == "det":
                    x_pred, curr_skip = self.model(x_j_black, m_in, r_j, a_j, skip)
                elif cf.model == "svg":
                    m_next_in = m_i
                    if cf.model_use_future_mask:
                        m_next_in = m_i.repeat(1, 2, 1, 1)
                    out = self.model(
                        x_j_black,
                        m_in,
                        r_in,
                        a_j,
                        x_i_black,
                        m_next_in,
                        r_i,
                        skip,
                        force_use_prior=True
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
                x_pred_black = self._zero_robot_region(true_mask[i], x_pred, False)
                x_i_black = self._zero_robot_region(true_mask[i], x_i, False)
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
        losses["gen_imgs"] = (255 * all_x_pred_black).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
        losses["true_imgs"] = (255 * all_x_i_black).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
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
    trainer = MultiRobotPredictionTrainer(config)
    trainer.train()
