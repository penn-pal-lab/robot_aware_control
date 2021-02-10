from collections import defaultdict
import logging
import os
from functools import partial

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from warnings import simplefilter # disable tensorflow warnings
simplefilter(action='ignore', category=FutureWarning)

import colorlog
import ipdb
import numpy as np
import torch
import wandb
from src.prediction.losses import dontcare_mse_criterion, kl_criterion, mse_criterion, l1_criterion, robot_mse_criterion, world_mse_criterion
from src.prediction.models.base import MLPEncoder, init_weights
from src.prediction.models.lstm import LSTM, GaussianLSTM
from src.utils.plot import save_gif, save_tensors_image
from torch import cat, optim
from tqdm import tqdm
from time import time
from math import floor
import imageio


class MultiRobotPredictionTrainer(object):
    """
    Video Prediction with multiple robot dataset
    Training, Checkpointing, Visualizing the prediction
    """

    def __init__(self, config):
        self._config = config
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print("using device for training", device)
        self._logger = colorlog.getLogger("file/console")

        self._device = device
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
            config_exclude_keys=["device"]
        )
        self._img_augmentation = config.img_augmentation

    def _init_models(self, cf):
        """Initialize models and optimizers
        When adding a new model, make sure to:
        - Call to(device)
        - Add optimizer
        - Add optimizer step() call
        - Update save and load ckpt code
        """
        self.all_models = []
        input_dim = cf.action_enc_dim + cf.robot_enc_dim + cf.g_dim
        if cf.stoch:
            input_dim += cf.z_dim
            self.posterior = post = GaussianLSTM(
                cf.robot_enc_dim + cf.g_dim,
                cf.z_dim,
                cf.rnn_size,
                cf.posterior_rnn_layers,
                cf.batch_size,
            ).to(self._device)

            self.prior = prior = GaussianLSTM(
                cf.action_enc_dim + cf.robot_enc_dim + cf.g_dim,
                cf.z_dim,
                cf.rnn_size,
                cf.prior_rnn_layers,
                cf.batch_size,
            ).to(self._device)
            self.all_models.extend([post, prior])

        self.frame_predictor = frame_pred = LSTM(
            input_dim,
            cf.g_dim,
            cf.rnn_size,
            cf.predictor_rnn_layers,
            cf.batch_size,
        ).to(self._device)

        if cf.image_width == 64:
            from src.prediction.models.vgg_64 import Decoder, Encoder
        elif cf.image_width == 128:
            from src.prediction.models.vgg import Decoder, Encoder

        # RGB + mask channel
        self.encoder = enc = Encoder(cf.g_dim, cf.channels + 1, cf.multiview, cf.dropout).to(
            self._device
        )
        self.decoder = dec = Decoder(cf.g_dim, cf.channels, cf.multiview).to(
            self._device
        )
        self.action_enc = ac = MLPEncoder(cf.action_dim, cf.action_enc_dim, 32).to(
            self._device
        )
        self.robot_enc = rob = MLPEncoder(cf.robot_dim, cf.robot_enc_dim, 32).to(
            self._device
        )

        self.all_models.extend([frame_pred, enc, dec, ac, rob])

        # initialize weights and count parameters
        # def count_parameters(model):
            # return sum(p.numel() for p in model.parameters() if p.requires_grad)
        # num_p = 0
        for model in self.all_models:
            model.apply(init_weights)
            # p = count_parameters(model)
            # print(model.name, p)
            # num_p += p
        # self._logger.info(f"total parameters: {num_p}")
        if cf.optimizer == "adam":
            optimizer = partial(optim.Adam, lr=cf.lr, betas=(cf.beta1, 0.999))
        elif cf.optimizer == "rmsprop":
            optimizer = optim.RMSprop
        elif cf.optimizer == "sgd":
            optimizer = optim.SGD
        else:
            raise ValueError("Unknown optimizer: %s" % cf.optimizer)

        self.frame_predictor_optimizer = optimizer(self.frame_predictor.parameters())
        if cf.stoch:
            self.posterior_optimizer = optimizer(self.posterior.parameters())
            self.prior_optimizer = optimizer(self.prior.parameters())
        self.encoder_optimizer = optimizer(self.encoder.parameters())
        self.decoder_optimizer = optimizer(self.decoder.parameters())
        self.action_encoder_optimizer = optimizer(self.action_enc.parameters())
        self.robot_encoder_optimizer = optimizer(self.robot_enc.parameters())


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
        robot_mask = robot_mask.repeat(1,3,1,1)
        image[robot_mask] *= 0

    def _train_step(self, data):
        """Forward and Backward pass of models
        Returns info dict containing loss metrics
        """
        cf = self._config
        for model in self.all_models:
            model.zero_grad()

        # initialize the recurrent states
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        if cf.stoch:
            self.posterior.hidden = self.posterior.init_hidden()
            self.prior.hidden = self.prior.init_hidden()

        losses = defaultdict(float)  # log loss metrics
        recon_loss = kld = 0
        (x, robot, ac, mask), robot_name = data
        robot_name = np.array(robot_name)
        all_robots = set(robot_name)
        x_pred = None
        for i in range(1, cf.n_past + cf.n_future):
            if i > 1:
                input_token = x[i - 1] if self._use_true_token() else x_pred.clone().detach()
            else:
                input_token = x[i - 1]
            # zero out robot pixels in input for norobot cost
            if self._config.reconstruction_loss == "dontcare_mse":
                self._zero_robot_region(mask[i-1], input_token)
                self._zero_robot_region(mask[i], x[i])
            h = self.encoder(cat([input_token, mask[i - 1]], dim=1))
            r = self.robot_enc(robot[i - 1])
            a = self.action_enc(ac[i - 1])
            h_target = self.encoder(cat([x[i], mask[i]], dim=1))[0]
            r_target = self.robot_enc(robot[i])
            # if n_past is 1, then we need to manually set skip var
            if (i == 1 and cf.n_past == 1) or cf.last_frame_skip or i < cf.n_past:
                h, skip = h
            else:
                h = h[0]

            if cf.stoch:
                z_t, mu, logvar = self.posterior(cat([r_target, h_target], 1))
                _, mu_p, logvar_p = self.prior(cat([a, r, h], 1))
                # use z_t from posterior for training stability.
                h_pred = self.frame_predictor(cat([a, r, h, z_t], 1))
            else:
                h_pred = self.frame_predictor(cat([a, r, h], 1))
            x_pred = self.decoder([h_pred, skip])  # N x C x H x W
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
                view_loss = self._recon_loss(x_pred, x[i], mask[i])
                recon_loss += view_loss # accumulate for backprop
                losses["recon_loss"] += view_loss.cpu().item()

                # for logging metrics
                with torch.no_grad():
                    robot_mse = robot_mse_criterion(x_pred, x[i], mask[i])
                    world_mse = world_mse_criterion(x_pred, x[i], mask[i])
                    losses["robot_loss"] += robot_mse.cpu().item()
                    losses["world_loss"] += world_mse.cpu().item()
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
                        losses[f"{r}_robot_loss"] += r_robot_mse.cpu().item()
                        losses[f"{r}_world_loss"] += r_world_mse.cpu().item()

            if cf.stoch:
                kl = kl_criterion(mu, logvar, mu_p, logvar_p, cf.batch_size)
                kld += kl
                losses["kld"] += kl.cpu().item()
        loss = recon_loss + kld * cf.beta
        loss.backward()
        self.frame_predictor_optimizer.step()
        if cf.stoch:
            self.posterior_optimizer.step()
            self.prior_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.action_encoder_optimizer.step()
        self.robot_encoder_optimizer.step()

        for k, v in losses.items():
            losses[k] = v / (cf.n_past + cf.n_future)
        return losses

    def _epoch_save_fid_images(self, random_snippet=False):
        """
        Save all model outputs and test set into a folder for FID calculations.
        """
        if random_snippet:
            self._snippet_rng = np.random.RandomState(self._config.seed)

        for i, (data, robot_name) in enumerate(tqdm(self.test_loader, "FID calculation")):
            # transpose from (B, L, C, W, H) to (L, B, C, W, H)
            frames, robots, actions, masks = data
            frames = frames.transpose_(1, 0).to(self._device)
            robots = robots.transpose_(1, 0).to(self._device)
            actions = actions.transpose_(1, 0).to(self._device)
            masks = masks.transpose_(1, 0).to(self._device)
            data = (frames, robots, actions, masks), robot_name
            self._video_save_fid(i, data, autoregressive=True, random_snippet=random_snippet)

    def _video_save_fid(self, idx,  data, autoregressive=False, random_snippet=False):
        """Evaluates over an entire video
        data: video data from dataloader
        autoregressive: use model's outputs as input for next timestep
        """
        (x, r, a, m), name = data
        T = len(x)
        window = self._config.n_past + self._config.n_future
        fid_folder = os.path.join(self._config.log_dir, "fid")
        target_folder = os.path.join(fid_folder, f"target_{self._step}")
        pred_folder = os.path.join(fid_folder, f"pred_{self._step}")
        os.makedirs(fid_folder, exist_ok=True)
        os.makedirs(target_folder, exist_ok=True)
        os.makedirs(pred_folder, exist_ok=True)
        if random_snippet:
            start = np.random.randint(0, floor(T/window))
            s = start * window
            e = (start+1) * window
            batch = (x[s:e], r[s:e], a[s:e-1], m[s:e]), name
            preds = self._get_preds_from_snippet(batch, autoregressive)
            # save the target images
            for j, imgs in enumerate(x[s+1:e]):
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
                    pred_img = (255 * pred_img.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
                    imageio.imwrite(pred_img_path, pred_img)
        else:
            for i in range(floor(T/window)):
                s = i * window
                e = (i+1) * window
                batch = (x[s:e], r[s:e], a[s:e-1], m[s:e]), name
                preds = self._get_preds_from_snippet(batch, autoregressive)
                # save the target images
                for j, imgs in enumerate(x[s+1:e]):
                    # imgs is (T x B x C x W x H), T is window length - 1
                    for t, img in enumerate(imgs):
                        img_name = f"vid_{idx}_snip_{s}_batch_{j}_time_{t}.png"
                        img_path = os.path.join(target_folder, img_name)
                        # convert image to H x W x C, [0,255] uint8
                        img = (255 * img.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
                        imageio.imwrite(img_path, img)

                        pred_img_name = f"vid_{idx}_snip_{s}_batch_{j}_time_{t}.png"
                        pred_img_path = os.path.join(target_folder, pred_img_name)
                        # ipdb.set_trace()
                        pred_img = preds[0][j][t]
                        pred_img = (255 * pred_img.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
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
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(bs)
        if cf.stoch:
            self.posterior.hidden = self.posterior.init_hidden(bs)
            self.prior.hidden = self.prior.init_hidden(bs)
        (x, robot, ac, mask), robot_name = data
        robot_name = np.array(robot_name)
        all_preds = defaultdict(list)
        x_pred = None
        for i in range(1, cf.n_past + cf.n_future):
            if autoregressive and i > 1:
                input_token = x_pred.clone().detach()
            else:
                input_token = x[i - 1]
            # zero out robot pixels in input for norobot cost
            if self._config.reconstruction_loss == "dontcare_mse":
                self._zero_robot_region(mask[i-1], input_token)
                self._zero_robot_region(mask[i], x[i])
            h = self.encoder(cat([input_token, mask[i - 1]], dim=1))
            r = self.robot_enc(robot[i - 1])
            a = self.action_enc(ac[i - 1])
            h_target = self.encoder(cat([x[i], mask[i]], dim=1))[0]
            r_target = self.robot_enc(robot[i])
            # if n_past is 1, then we need to manually set skip var
            if (i == 1 and cf.n_past == 1) or cf.last_frame_skip or i < cf.n_past:
                h, skip = h
            else:
                h = h[0]

            if cf.stoch:
                z_t, mu, logvar = self.posterior(cat([r_target, h_target], 1))
                _, mu_p, logvar_p = self.prior(cat([a, r, h], 1))
                h_pred = self.frame_predictor(cat([a, r, h, z_t], 1))
            else:
                h_pred = self.frame_predictor(cat([a, r, h], 1))
            x_pred = self.decoder([h_pred, skip])
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


    def _eval_epoch(self):
        losses = defaultdict(list)
        for data, robot_name in tqdm(self.test_loader, "evaluating epoch"):
            # transpose from (B, L, C, W, H) to (L, B, C, W, H)
            frames, robots, actions, masks = data
            frames = frames.transpose_(1, 0).to(self._device)
            robots = robots.transpose_(1, 0).to(self._device)
            actions = actions.transpose_(1, 0).to(self._device)
            masks = masks.transpose_(1, 0).to(self._device)
            data = (frames, robots, actions, masks), robot_name
            info = self._eval_video(data)
            for k, v in info.items():
                losses[k].append(v)
            info = self._eval_video(data, autoregressive=True)
            for k, v in info.items():
                losses[k].append(v)

        avg_loss = {f"test/{k}": np.mean(v) for k, v in losses.items()}
        # epoch_loss = {f"test/epoch_{k}": np.sum(v) for k, v in losses.items()}
        log_str = ""
        for k, v in avg_loss.items():
            log_str += f"{k}: {v:.5f}, "
        self._logger.info(log_str)
        wandb.log(avg_loss, step=self._step)

    def _eval_transfer(self):
        losses = defaultdict(list)
        for data, robot_name in tqdm(self.transfer_loader, "evaluating transfer"):
            # transpose from (B, L, C, W, H) to (L, B, C, W, H)
            frames, robots, actions, masks = data
            frames = frames.transpose_(1, 0).to(self._device)
            robots = robots.transpose_(1, 0).to(self._device)
            actions = actions.transpose_(1, 0).to(self._device)
            masks = masks.transpose_(1, 0).to(self._device)
            data = (frames, robots, actions, masks), robot_name
            info = self._eval_video(data)
            for k, v in info.items():
                losses[k].append(v)
            info = self._eval_video(data, autoregressive=True)
            for k, v in info.items():
                losses[k].append(v)
        avg_loss = {f"transfer/{k}": np.mean(v) for k, v in losses.items()}
        log_str = ""
        for k, v in avg_loss.items():
            log_str += f"{k}: {v:.5f}, "
        self._logger.info(log_str)
        wandb.log(avg_loss, step=self._step)

    def _eval_video(self, data, autoregressive=False):
        """Evaluates over an entire video
        data: video data from dataloader
        autoregressive: use model's outputs as input for next timestep
        """
        (x, r, a, m), name = data
        T = len(x)
        window = self._config.n_past + self._config.n_future
        all_losses = defaultdict(float)
        for i in range(floor(T/window)):
            s = i * window
            e = (i+1) * window
            batch = (x[s:e], r[s:e], a[s:e-1], m[s:e]), name
            losses = self._eval_step(batch, autoregressive)
            for k, v in losses.items():
                all_losses[k] += v / floor(T/window)
        return all_losses

    @torch.no_grad()
    def _eval_step(self, data, autoregressive=False):
        """
        Evals over a snippet of video of length n_past + n_future
        autoregressive: use model's outputs as input for next timestep
        """
        # one step evaluation loss
        cf = self._config
        (x, robot, ac, mask), robot_name = data
        bs = min(cf.test_batch_size, x.shape[1])
        # initialize the recurrent states
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(bs)
        if cf.stoch:
            self.posterior.hidden = self.posterior.init_hidden(bs)
            self.prior.hidden = self.prior.init_hidden(bs)

        losses = defaultdict(float)
        robot_name = np.array(robot_name)
        all_robots = set(robot_name)
        x_pred = None
        prefix = "autoreg" if autoregressive else "1step"
        for i in range(1, cf.n_past + cf.n_future):
            if autoregressive and i > 1:
                input_token = x_pred.clone().detach()
            else:
                input_token = x[i - 1]
            # zero out robot pixels in input for norobot cost
            if self._config.reconstruction_loss == "dontcare_mse":
                self._zero_robot_region(mask[i-1], input_token)
                self._zero_robot_region(mask[i], x[i])
            h = self.encoder(cat([input_token, mask[i - 1]], dim=1))
            r = self.robot_enc(robot[i - 1])
            a = self.action_enc(ac[i - 1])
            h_target = self.encoder(cat([x[i], mask[i]], dim=1))[0]
            r_target = self.robot_enc(robot[i])
            # if n_past is 1, then we need to manually set skip var
            if (i == 1 and cf.n_past == 1) or cf.last_frame_skip or i < cf.n_past:
                h, skip = h
            else:
                h = h[0]

            if cf.stoch:
                z_t, mu, logvar = self.posterior(cat([r_target, h_target], 1))
                _, mu_p, logvar_p = self.prior(cat([a, r, h], 1))
                h_pred = self.frame_predictor(cat([a, r, h, z_t], 1))
            else:
                h_pred = self.frame_predictor(cat([a, r, h], 1))
            x_pred = self.decoder([h_pred, skip])
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

            if cf.stoch:
                kl = kl_criterion(mu, logvar, mu_p, logvar_p, cf.batch_size)
                losses[f"{prefix}_kld"] += kl.cpu().item()

        for k, v in losses.items():
            losses[k] = v / (cf.n_past + cf.n_future)
        return losses

    def train(self):
        """Training, Evaluation, Checkpointing loop"""
        cf = self._config
        # load models and dataset
        self._step = self._load_checkpoint(cf.dynamics_model_ckpt)
        self._setup_data()

        # start training
        progress = tqdm(initial=self._step, total=cf.niter * cf.epoch_size)
        for epoch in range(cf.niter):
            for model in self.all_models:
                model.train()
            # epoch_losses = defaultdict(float)
            for i in range(cf.epoch_size):
                # start = time()
                data = next(self.training_batch_generator)
                # end = time()
                # data_time = end - start
                # print("data loading time", data_time)

                # start = time()
                # with torch.autograd.set_detect_anomaly(True):
                info = self._train_step(data)
                # end = time()
                # update_time = end - start
                # print("network update time", update_time)
                # for k, v in info.items():
                #     epoch_losses[f"train/epoch_{k}"] += v
                info["sample_schedule"] = self._schedule_prob()[0]
                self._step += 1

                if i == cf.epoch_size - 1:
                    self.plot(data, epoch, "train")
                    self.plot_rec(data, epoch, "train")

                wandb.log({f"train/{k}": v for k, v in info.items()}, step=self._step)
                progress.update()

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
                for model in self.all_models:
                    model.eval()
                self._eval_epoch()
                test_data = next(self.testing_batch_generator)
                self.plot(test_data, epoch, "test")
                self.plot_rec(test_data, epoch, "test")
                comp_data = next(self.comp_batch_generator)
                self.plot(comp_data, epoch, "comparison")

                if self._config.training_regime == "singlerobot":
                    self._eval_transfer()
                    transfer_data = next(self.transfer_batch_generator)
                    self.plot(transfer_data, epoch, "transfer")

    def _save_checkpoint(self):
        path = os.path.join(self._config.log_dir, f"ckpt_{self._step}.pt")
        data = {
            "encoder": self.encoder,
            "robot_enc": self.robot_enc,
            "action_enc": self.action_enc,
            "decoder": self.decoder,
            "frame_predictor": self.frame_predictor,
            "step": self._step,
        }
        if self._config.stoch:
            data.update({"posterior": self.posterior, "prior": self.prior})
        torch.save(data, path)

    def _load_checkpoint(self, ckpt_path=None):
        """
        Either load a given checkpoint path, or find the most recent checkpoint file
        in the log dir and start from there

        Returns the training step
        """

        def load_models(ckpt):
            self.frame_predictor = ckpt["frame_predictor"]
            if self._config.stoch:
                self.posterior = ckpt["posterior"]
                self.prior = ckpt["prior"]
            self.decoder = ckpt["decoder"]
            self.encoder = ckpt["encoder"]
            self.robot_enc = ckpt["robot_enc"]
            self.action_enc = ckpt["action_enc"]

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
            ckpt_path, ckpt_num = get_recent_ckpt_path(self._config.log_dir)
            if ckpt_path is None:
                print("Randomly initializing Model")
                return 0
            else:
                print(f"Loading most recent ckpt: {ckpt_path}")
                ckpt = torch.load(ckpt_path)
                load_models(ckpt)
                step = ckpt["step"]
                return step
        else:
            print(f"Loading ckpt {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            load_models(ckpt)
            step = ckpt["step"]
            if self._config.training_regime == "finetune":
                step = 0
            return step

    def _setup_data(self):
        """
        Setup the dataset and dataloaders
        """
        if self._config.training_regime == "multirobot":
            from src.dataset.multirobot_dataloaders import create_loaders, get_batch
        elif self._config.training_regime == "singlerobot":
            from src.dataset.finetune_multirobot_dataloaders import create_loaders, get_batch, create_transfer_loader
            # measure zero shot performance on transfer data
            self.transfer_loader = create_transfer_loader(self._config)
            self.transfer_batch_generator = get_batch(self.transfer_loader, self._device)
        elif self._config.training_regime == "finetune":
            from src.dataset.finetune_multirobot_dataloaders import create_finetune_loaders as create_loaders, get_batch
        else:
            raise NotImplementedError(self._config.training_regime)
        train_loader, self.test_loader, comp_loader = create_loaders(self._config)
         # for infinite batching
        self.training_batch_generator = get_batch(train_loader, self._device)
        self.testing_batch_generator = get_batch(self.test_loader, self._device)
        self.comp_batch_generator = get_batch(comp_loader, self._device)

    @torch.no_grad()
    def plot(self, data, epoch, name):
        """
        Plot the generation with learned prior. Autoregressive output.
        """
        cf = self._config
        (x, robot, ac, mask), _ = data
        nsample = 1
        gen_seq = [[] for i in range(nsample)]
        gt_seq = [x[i] for i in range(len(x))]
        b = min(x.shape[1], 10)
        # truncate batch
        x = x[:, :b]
        robot = robot[:, :b]
        ac = ac[:, :b]
        mask = mask[:, :b]
        for s in range(nsample):
            self.frame_predictor.hidden = self.frame_predictor.init_hidden(b)
            if self._config.stoch:
                self.posterior.hidden = self.posterior.init_hidden(b)
                self.prior.hidden = self.prior.init_hidden(b)
            # first frame of all videos
            if self._config.reconstruction_loss == "dontcare_mse":
                self._zero_robot_region(mask[0], x[0])
            gen_seq[s].append(x[0])
            x_in = x[0]
            for i in range(1, cf.n_eval):
                # zero out robot pixels in input for norobot cost
                if self._config.reconstruction_loss == "dontcare_mse":
                    self._zero_robot_region(mask[i], x[i])
                h = self.encoder(cat([x_in, mask[i - 1]], dim=1))
                r = self.robot_enc(robot[i - 1])
                a = self.action_enc(ac[i - 1])
                if (i == 1 and cf.n_past == 1) or cf.last_frame_skip or i < cf.n_past:
                    h, skip = h
                else:
                    h, _ = h
                h = h.detach()
                if i < cf.n_past:
                    r_target = self.robot_enc(robot[i])
                    h_target = self.encoder(cat([x[i], mask[i]], dim=1))
                    h_target = h_target[0]
                    if cf.stoch:
                        z_t, _, _ = self.posterior(cat([r_target, h_target], 1))
                        # condition the recurrent state of prior
                        self.prior(cat([a, r, h], 1))
                        self.frame_predictor(cat([a, r, h, z_t], 1))
                    else:
                        self.frame_predictor(cat([a, r, h], 1))
                    x_in = x[i]
                    gen_seq[s].append(x_in)
                else:
                    if cf.stoch:
                        z_t, _, _ = self.prior(cat([a, r, h], 1))
                        h = self.frame_predictor(cat([a, r, h, z_t], 1))
                    else:
                        h = self.frame_predictor(cat([a, r, h], 1))
                    x_in = self.decoder([h, skip])
                    if self._config.reconstruction_loss == "dontcare_mse":
                        self._zero_robot_region(mask[i], x_in)
                    gen_seq[s].append(x_in)

        to_plot = []
        gifs = [[] for t in range(cf.n_eval)]
        nrow = b
        for i in range(nrow):
            # ground truth sequence
            row = []
            for t in range(cf.n_eval):
                row.append(gt_seq[t][i])
            to_plot.append(row)
            if self._config.stoch:
                # best sequence
                min_mse = 1e7
                for s in range(nsample):
                    mse = 0
                    for t in range(cf.n_eval):
                        mse += torch.sum(
                            (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu()) ** 2
                        )
                    if mse < min_mse:
                        min_mse = mse
                        min_idx = s
                    s_list = [
                        min_idx,
                        np.random.randint(nsample),
                        np.random.randint(nsample),
                        np.random.randint(nsample),
                        np.random.randint(nsample),
                    ]
            else:
                s_list = [0]
            for ss in range(len(s_list)):
                s = s_list[ss]
                row = []
                for t in range(cf.n_eval):
                    row.append(gen_seq[s][t][i])
                to_plot.append(row)
            for t in range(cf.n_eval):
                row = []
                row.append(gt_seq[t][i])
                for ss in range(len(s_list)):
                    s = s_list[ss]
                    row.append(gen_seq[s][t][i])
                gifs[t].append(row)

        fname = os.path.join(cf.plot_dir, f"{name}_{epoch}.png")
        save_tensors_image(fname, to_plot)

        fname = os.path.join(cf.plot_dir, f"{name}_{epoch}.gif")
        save_gif(fname, gifs)
        wandb.log({f"{name}/gifs": wandb.Video(fname, format="gif")}, step=self._step)

    @torch.no_grad()
    def plot_rec(self, data, epoch, name):
        """
        Plot the 1 step reconstruction with posterior instead of learned prior
        """
        cf = self._config
        (x, robot, ac, mask), _ = data
        b = min(x.shape[1], 10)
        # truncate batch
        x = x[:, :b]
        robot = robot[:, :b]
        ac = ac[:, :b]
        mask = mask[:, :b]

        self.frame_predictor.hidden = self.frame_predictor.init_hidden(b)
        if cf.stoch:
            self.posterior.hidden = self.posterior.init_hidden(b)
        gen_seq = []
        if self._config.reconstruction_loss == "dontcare_mse":
            self._zero_robot_region(mask[0], x[0])
        gen_seq.append(x[0])
        for i in range(1, cf.n_past + cf.n_future):
            # zero out robot pixels in input for norobot cost
            if self._config.reconstruction_loss == "dontcare_mse":
                self._zero_robot_region(mask[i], x[i])
            h = self.encoder(cat([x[i - 1], mask[i - 1]], dim=1))
            r = self.robot_enc(robot[i - 1])
            a = self.action_enc(ac[i - 1])
            h_target = self.encoder(cat([x[i], mask[i]], dim=1))[0]
            r_target = self.robot_enc(robot[i])
            if (i == 1 and cf.n_past == 1) or cf.last_frame_skip or i < cf.n_past:
                h, skip = h
            else:
                h, _ = h
            if cf.stoch:
                z_t, _, _ = self.posterior(cat([r_target, h_target], 1))
                embed = cat([a, r, h, z_t], 1)
            else:
                embed = cat([a, r, h], 1)

            if i < cf.n_past:
                self.frame_predictor(embed)
                gen_seq.append(x[i])
            else:
                h_pred = self.frame_predictor(embed)
                x_pred = self.decoder([h_pred, skip]).detach()
                if self._config.reconstruction_loss == "dontcare_mse":
                    self._zero_robot_region(mask[i], x_pred)
                gen_seq.append(x_pred)

        to_plot = []
        nrow = min(cf.batch_size, 10)
        for i in range(nrow):
            row = []
            for t in range(cf.n_past + cf.n_future):
                row.append(gen_seq[t][i])
            to_plot.append(row)
        fname = os.path.join(cf.plot_dir, f"{name}_rec_{epoch}.png")
        save_tensors_image(fname, to_plot)


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

    # mp.set_start_method("spawn")
    config, _ = argparser()
    make_log_folder(config)
    trainer = MultiRobotPredictionTrainer(config)
    trainer.train()
