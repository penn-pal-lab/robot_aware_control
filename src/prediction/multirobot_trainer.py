import logging
import os
from collections import defaultdict
from functools import partial

from src.dataset.multirobot_dataset import get_batch, process_batch
from src.prediction.models.dynamics import CopyModel, DeterministicModel, SVGModel

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
    dontcare_mse_criterion,
    kl_criterion,
    l1_criterion,
    mse_criterion,
    robot_mse_criterion,
    world_mse_criterion,
)
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
        os.environ["WANDB_API_KEY"] = "24e6ba2cb3e7bced52962413c58277801d14bba0"
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

    def _init_models(self, cf):
        """Initialize models and optimizers
        When adding a new model, make sure to:
        - Call to(device)
        - Add optimizer
        - Add optimizer step() call
        - Update save and load ckpt code
        """
        if cf.model == "svg":
            self.model = SVGModel(cf).to(self._device)
        elif cf.model == "det":
            self.model = DeterministicModel(cf).to(self._device)
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

        self.optimizer = optimizer(self.model.parameters())

    def _schedule_prob(self):
        """Returns probability of using ground truth"""
        # assume 400k max training steps
        # https://www.desmos.com/calculator/bo4aoyqje1
        k = 10000
        use_truth = k / (k + np.exp(self._step / 3900))
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
        else:
            raise NotImplementedError(f"{self._config.reconstruction_loss}")

    def _zero_robot_region(self, mask, image):
        """
        Set the robot region to zero
        """
        robot_mask = mask.type(torch.bool)
        robot_mask = robot_mask.repeat(1, 3, 1, 1)
        image[robot_mask] *= 0

    def _train_video(self, data):
        """Train the model over the video data

        Slices video up into K length sequences for training.
        Args:
            data (dict): Video data
        """
        x = data["images"]
        T = len(x)
        window = self._config.n_past + self._config.n_future
        all_losses = defaultdict(float)
        for i in range(floor(T / window)):
            s = i * window
            e = (i + 1) * window
            batch_data = {
                "images": x[s:e],
                "states": data["states"][s:e],
                "actions": data["actions"][s : e - 1],
                "masks": data["masks"][s:e],
                "robot": data["robot"],
            }
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
        robot = data["states"]
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
        for i in range(1, cf.n_past + cf.n_future):
            if i > 1:
                x_j = x[i - 1] if self._use_true_token() else x_pred.clone().detach()
            else:
                x_j = x[i - 1]
            # let j be i - 1, or previous timestep
            m_j, r_j, a_j = mask[i - 1], robot[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], mask[i], robot[i]

            # zero out robot pixels in input for norobot cost
            if self._config.reconstruction_loss == "dontcare_mse":
                self._zero_robot_region(m_j, x_j)
                self._zero_robot_region(m_i, x_i)
            if cf.model == "det":
                x_pred, curr_skip = self.model(x_j, m_j, r_j, a_j, skip)
            elif cf.model == "svg":
                out = self.model(x_j, m_j, r_j, a_j, x_i, m_i, r_i, skip)
                x_pred, curr_skip, mu, logvar, mu_p, logvar_p = out
            # overwrite skip with most recent skip
            if cf.last_frame_skip or i <= cf.n_past:
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
            losses[k] = v / (cf.n_past + cf.n_future)
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
            info = self._eval_video(data)
            for k, v in info.items():
                losses[k].append(v)
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
        x = data["images"]
        T = len(x)
        window = self._config.n_eval
        all_losses = defaultdict(float)
        for i in range(floor(T / window)):
            s = i * window
            e = (i + 1) * window
            batch_data = {
                "images": x[s:e],
                "states": data["states"][s:e],
                "actions": data["actions"][s : e - 1],
                "masks": data["masks"][s:e],
                "robot": data["robot"],
            }
            losses = self._eval_step(batch_data, autoregressive)
            for k, v in losses.items():
                all_losses[k] += v
        for k, v in all_losses.items():
            all_losses[k] /= floor(T/window)
        return all_losses

    @torch.no_grad()
    def _eval_step(self, data, autoregressive=False):
        """
        Evals over a snippet of video of length n_past + n_future
        autoregressive: use model's outputs as input for next timestep
        """
        # one step evaluation loss
        cf = self._config

        x = data["images"]
        robot = data["states"]
        ac = data["actions"]
        mask = data["masks"]
        robot_name = data["robot"]
        bs = min(cf.test_batch_size, x.shape[1])
        # initialize the recurrent states
        self.model.init_hidden(bs)

        losses = defaultdict(float)
        robot_name = np.array(robot_name)
        all_robots = set(robot_name)
        x_pred = skip = None
        prefix = "autoreg" if autoregressive else "1step"
        for i in range(1, cf.n_eval):
            if autoregressive and i > 1:
                x_j = x_pred.clone().detach()
            else:
                x_j = x[i - 1]
            # let j be i - 1, or previous timestep
            m_j, r_j, a_j = mask[i - 1], robot[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], mask[i], robot[i]

            if cf.model == "copy":
                x_pred = self.model(x_j, m_j, x_i, m_i)
            else:
                # zero out robot pixels in input for norobot cost
                if self._config.reconstruction_loss == "dontcare_mse":
                    self._zero_robot_region(m_j, x_j)
                    self._zero_robot_region(m_i, x_i)

                if cf.model == "det":
                    x_pred, curr_skip = self.model(x_j, m_j, r_j, a_j, skip)
                elif cf.model == "svg":
                    # use prior for autoregressive step and i > conditioning
                    force_use_prior = autoregressive and i > 1
                    out = self.model(x_j, m_j, r_j, a_j, x_i, m_i, r_i, skip, force_use_prior=force_use_prior)
                    x_pred, curr_skip, mu, logvar, mu_p, logvar_p = out

                # overwrite skip with most recent skip
                if cf.last_frame_skip or i <= cf.n_past:
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
                view_loss = self._recon_loss(x_pred, x[i], mask[i])
                losses[f"{prefix}_recon_loss"] += view_loss.cpu().item()
                robot_mse = robot_mse_criterion(x_pred, x[i], mask[i])
                world_mse = world_mse_criterion(x_pred, x[i], mask[i])
                losses[f"{prefix}_robot_loss"] += robot_mse.cpu().item()
                losses[f"{prefix}_world_loss"] += world_mse.cpu().item()
                # robot specific metrics
                for r in all_robots:
                    if len(all_robots) == 1:
                        break
                    r_idx = r == robot_name
                    r_pred = x_pred[r_idx]
                    r_img = x[i][r_idx]
                    r_mask = mask[i][r_idx]
                    r_robot_mse = robot_mse_criterion(r_pred, r_img, r_mask)
                    r_world_mse = world_mse_criterion(r_pred, r_img, r_mask)
                    losses[f"{prefix}_{r}_robot_loss"] += r_robot_mse.cpu().item()
                    losses[f"{prefix}_{r}_world_loss"] += r_world_mse.cpu().item()

            if cf.model == "svg":
                # if x.shape[1] != cf.test_batch_size:
                #     ipdb.set_trace()
                kl = kl_criterion(mu, logvar, mu_p, logvar_p, bs)
                losses[f"{prefix}_kld"] += kl.cpu().item()

        for k, v in losses.items():
            losses[k] = v / cf.n_eval
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
        total = cf.niter * cf.epoch_size
        desc = "batches seen"
        self.progress = tqdm(initial=self._step, total=total, desc=desc)

        # start training
        for epoch in range(cf.niter):
            self.model.train()
            # epoch_losses = defaultdict(float)
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
                self._step += 1

                if i == cf.epoch_size - 1:
                    self.plot(data, epoch, "train")
                    self.plot_rec(data, epoch, "train")

                wandb.log({f"train/{k}": v for k, v in info.items()}, step=self._step)
                self.progress.update()

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
            if epoch % cf.eval_interval == 0:
                # plot and evaluate on test set
                self.model.eval()
                info = self._compute_epoch_metrics(self.test_loader, "test")
                wandb.log(info, step=self._step)
                test_data = next(self.testing_batch_generator)
                self.plot(test_data, epoch, "test")
                self.plot_rec(test_data, epoch, "test")
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
        self.plot_rec(train_data, epoch, "train")
        # test set
        self.model.eval()
        test_info = self._compute_epoch_metrics(self.test_loader, "test")
        test_data = next(self.testing_batch_generator)
        self.plot(test_data, epoch, "test")
        self.plot_rec(test_data, epoch, "test")

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
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
        }
        torch.save(data, path)

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
                self.model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                step = ckpt["step"]
                return step
        else:
            # load given ckpt path
            print(f"Loading ckpt {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self._device)
            self.model.load_state_dict(ckpt["model"])
            if self._config.training_regime in ["finetune", "finetune_sawyer_view"]:
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
            from src.dataset.finetune_multirobot_dataloaders import (
                create_loaders,
                create_transfer_loader,
            )

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
        robot = data["states"]
        ac = data["actions"]
        mask = data["masks"]
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
        robot = robot[start:end, :b]
        ac = ac[start: end - 1, :b]
        mask = mask[start:end, :b]
        gen_seq = [[] for i in range(nsample)]
        gt_seq = [x[i] for i in range(len(x))]

        skip = None
        for s in range(nsample):
            self.model.init_hidden(b)
            if cf.reconstruction_loss == "dontcare_mse" and cf.model != "copy":
                self._zero_robot_region(mask[0], x[0])
            gen_seq[s].append(x[0])

            x_j = x[0]
            for i in range(1, video_len):
                # let j be i - 1, or previous timestep
                m_j, r_j, a_j = mask[i - 1], robot[i - 1], ac[i - 1]
                x_i, m_i, r_i = x[i], mask[i], robot[i]
                if cf.model == "copy":
                    x_pred = self.model(x_j, m_j, x_i, m_i)
                else:
                    # zero out robot pixels in input for norobot cost
                    if cf.reconstruction_loss == "dontcare_mse":
                        self._zero_robot_region(mask[i], x[i])

                    if cf.model == "det":
                        x_pred, curr_skip = self.model(x_j, m_j, r_j, a_j, skip)
                    elif cf.model == "svg":
                        if i > cf.n_past:  # don't use posterior
                            x_i, m_i, r_i = None, None, None
                        out = self.model(x_j, m_j, r_j, a_j, x_i, m_i, r_i, skip)
                        x_pred, curr_skip, _, _, _, _ = out

                    if cf.last_frame_skip or i <= cf.n_past:
                        # feed in the  most recent conditioning frame img's skip
                        skip = curr_skip

                    if cf.reconstruction_loss == "dontcare_mse":
                        self._zero_robot_region(mask[i], x_pred)
                if i < cf.n_past:
                    x_j = x_i
                else:
                    x_j = x_pred
                gen_seq[s].append(x_j)

        to_plot = []
        gifs = [[] for t in range(video_len)]
        nrow = b
        for i in range(nrow):
            # ground truth sequence
            row = []
            for t in range(video_len):
                row.append(gt_seq[t][i])
            to_plot.append(row)
            if cf.model == "svg":
                s_list = range(nsample)
            else:
                s_list = [0]
            for ss in range(len(s_list)):
                s = s_list[ss]
                row = []
                for t in range(video_len):
                    row.append(gen_seq[s][t][i])
                to_plot.append(row)
            for t in range(video_len):
                row = []
                row.append(gt_seq[t][i])
                for ss in range(len(s_list)):
                    s = s_list[ss]
                    row.append(gen_seq[s][t][i])
                gifs[t].append(row)
        # gifs is T x B x S x |I|
        fname = os.path.join(cf.plot_dir, f"{name}_{epoch}.png")
        save_tensors_image(fname, to_plot)

        fname = os.path.join(cf.plot_dir, f"{name}_{epoch}.gif")
        save_gif(fname, gifs)
        wandb.log({f"{name}/gifs": wandb.Video(fname, format="gif")}, step=self._step)

    @torch.no_grad()
    def plot_rec(self, data, epoch, name, random_start=True):
        """
        Plot the 1 step reconstruction with posterior instead of learned prior
        """
        cf = self._config
        x = data["images"]
        robot = data["states"]
        ac = data["actions"]
        mask = data["masks"]
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
        robot = robot[start:end, :b]
        ac = ac[start: end - 1, :b]
        mask = mask[start:end, :b]

        self.model.init_hidden(b)
        gen_seq = []
        if cf.reconstruction_loss == "dontcare_mse" and cf.model != "copy":
            self._zero_robot_region(mask[0], x[0])
        gen_seq.append(x[0])
        skip = None
        for i in range(1, video_len):
            # let j be i - 1, or previous timestep
            x_j, m_j, r_j, a_j = x[i - 1], mask[i - 1], robot[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], mask[i], robot[i]

            if cf.model == "copy":
                x_pred = self.model(x_j, m_j, x_i, m_i)
            else:
                # zero out robot pixels in input for norobot cost
                if cf.reconstruction_loss == "dontcare_mse":
                    self._zero_robot_region(mask[i], x[i])

                if cf.model == "det":
                    x_pred, curr_skip = self.model(x_j, m_j, r_j, a_j, skip)
                elif cf.model == "svg":
                    out = self.model(x_j, m_j, r_j, a_j, x_i, m_i, r_i, skip)
                    x_pred, curr_skip, _, _, _, _ = out
                # overwrite skip with most recent skip
                if cf.last_frame_skip or i <= cf.n_past:
                    skip = curr_skip
                if cf.reconstruction_loss == "dontcare_mse":
                    self._zero_robot_region(mask[i], x_pred)
            if i < cf.n_past:
                gen_seq.append(x_i)
            else:
                gen_seq.append(x_pred)

        to_plot = []
        nrow = b
        for i in range(nrow):
            row = []
            for t in range(video_len):
                row.append(gen_seq[t][i])
            to_plot.append(row)
        fname = os.path.join(cf.plot_dir, f"{name}_rec_{epoch}.png")
        save_tensors_image(fname, to_plot)

    def _epoch_save_fid_images(self, random_snippet=False):
        """
        Save all model outputs and test set into a folder for FID calculations.
        """
        if random_snippet:
            self._snippet_rng = np.random.RandomState(self._config.seed)

        for i, (data, robot_name) in enumerate(
            tqdm(self.test_loader, "FID calculation")
        ):
            self._video_save_fid(
                i, data, autoregressive=True, random_snippet=random_snippet
            )

    def _video_save_fid(self, idx, data, autoregressive=False, random_snippet=False):
        """Evaluates over an entire video
        data: video data from dataloader
        autoregressive: use model's outputs as input for next timestep
        """
        x = data["images"]
        r = data["states"]
        a = data["actions"]
        m = data["masks"]
        name = data["robot"]
        T = len(x)
        window = self._config.n_past + self._config.n_future
        fid_folder = os.path.join(self._config.log_dir, "fid")
        target_folder = os.path.join(fid_folder, f"target_{self._step}")
        pred_folder = os.path.join(fid_folder, f"pred_{self._step}")
        os.makedirs(fid_folder, exist_ok=True)
        os.makedirs(target_folder, exist_ok=True)
        os.makedirs(pred_folder, exist_ok=True)
        if random_snippet:
            start = np.random.randint(0, floor(T / window))
            s = start * window
            e = (start + 1) * window
            batch = (x[s:e], r[s:e], a[s : e - 1], m[s:e]), name
            preds = self._get_preds_from_snippet(batch, autoregressive)
            # save the target images
            for j, imgs in enumerate(x[s + 1 : e]):
                # imgs is (T x B x C x W x H), T is window length - 1
                for t, img in enumerate(imgs):
                    img_name = f"vid_{idx}_snip_{s}_batch_{j}_time_{t}.png"
                    img_path = os.path.join(target_folder, img_name)
                    # convert image to H x W x C, [0,255] uint8
                    img = (255 * img.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
                    imageio.imwrite(img_path, img)

                    pred_img_name = f"vid_{idx}_snip_{s}_batch_{j}_time_{t}.png"
                    pred_img_path = os.path.join(pred_folder, pred_img_name)
                    # ipdb.set_trace()
                    pred_img = preds[0][j][t]
                    pred_img = (255 * pred_img.permute(1, 2, 0).cpu().numpy()).astype(
                        np.uint8
                    )
                    imageio.imwrite(pred_img_path, pred_img)
        else:
            for i in range(floor(T / window)):
                s = i * window
                e = (i + 1) * window
                batch = (x[s:e], r[s:e], a[s : e - 1], m[s:e]), name
                preds = self._get_preds_from_snippet(batch, autoregressive)
                # save the target images
                for j, imgs in enumerate(x[s + 1 : e]):
                    # imgs is (T x B x C x W x H), T is window length - 1
                    for t, img in enumerate(imgs):
                        img_name = f"vid_{idx}_snip_{s}_batch_{j}_time_{t}.png"
                        img_path = os.path.join(target_folder, img_name)
                        # convert image to H x W x C, [0,255] uint8
                        img = (255 * img.permute(1, 2, 0).cpu().numpy()).astype(
                            np.uint8
                        )
                        imageio.imwrite(img_path, img)

                        pred_img_name = f"vid_{idx}_snip_{s}_batch_{j}_time_{t}.png"
                        pred_img_path = os.path.join(target_folder, pred_img_name)
                        # ipdb.set_trace()
                        pred_img = preds[0][j][t]
                        pred_img = (
                            255 * pred_img.permute(1, 2, 0).cpu().numpy()
                        ).astype(np.uint8)
                        imageio.imwrite(pred_img_path, pred_img)

    @torch.no_grad()
    def _get_preds_from_snippet(self, data, autoregressive=False):
        """
        Gets model predictions from a snippet of video of length n_past + n_future
        autoregressive: use model's outputs as input for next timestep
        Returns a dictionary where key is viewpoint, value is model outputs
        """
        # one step evaluation loss
        cf = self._config
        bs = cf.test_batch_size
        # initialize the recurrent states
        self.model.init_hidden(bs)
        x = data["images"]
        robot = data["states"]
        ac = data["actions"]
        mask = data["masks"]
        robot_name = data["robot"]
        robot_name = np.array(robot_name)
        all_preds = defaultdict(list)
        x_pred = None
        for i in range(1, cf.n_past + cf.n_future):
            if autoregressive and i > 1:
                x_j = x_pred.clone().detach()
            else:
                x_j = x[i - 1]
            # let j be i - 1, or previous timestep
            m_j, r_j, a_j = mask[i - 1], robot[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], mask[i], robot[i]

            # zero out robot pixels in input for norobot cost
            if self._config.reconstruction_loss == "dontcare_mse":
                self._zero_robot_region(m_j, x_j)
                self._zero_robot_region(m_i, x_i)

            if cf.model == "det":
                x_pred, curr_skip = self.model(x_j, m_j, r_j, a_j, skip)
            elif cf.model == "svg":
                out = self.model(x_j, m_j, r_j, a_j, x_i, m_i, r_i, skip)
                x_pred, curr_skip, _, _, _, _ = out
            # overwrite skip with most recent skip
            if cf.last_frame_skip or i <= cf.n_past:
                skip = curr_skip

            if cf.multiview:
                num_views = x_pred.shape[2] // cf.image_width
                for n in range(num_views):
                    start, end = n * cf.image_width, (n + 1) * cf.image_width
                    view_pred = x_pred[:, :, start:end, :]
                    # view = x[i][:, :, start:end, :]
                    # view_mask = mask[i][:, :, start:end, :]
                    all_preds[n].append(view_pred)
            else:
                all_preds[0].append(x_pred)
        return all_preds


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