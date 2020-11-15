import logging
import os
from functools import partial

import colorlog
import ipdb
import numpy as np
import torch
import wandb
from src.dataset.dataloaders import create_loaders, get_batch
from src.prediction.losses import kl_criterion, mse_criterion
from src.prediction.models.base import MLPEncoder, init_weights
from src.prediction.models.lstm import LSTM, GaussianLSTM
from src.utils.plot import save_gif, save_tensors_image
from torch import cat, optim
from tqdm import tqdm


class PredictionTrainer(object):
    """Training, Checkpointing, Visualizing the prediction"""

    def __init__(self, config):
        self._config = config
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self._device = device
        self._init_models(config)

        # init WandB
        if not config.wandb:
            os.environ["WANDB_MODE"] = "dryrun"
        os.environ["WANDB_API_KEY"] = "24e6ba2cb3e7bced52962413c58277801d14bba0"
        exclude = ["device"]
        wandb.init(
            resume=config.jobname,
            project=config.wandb_project,
            config={k: v for k, v in config.__dict__.items() if k not in exclude},
            dir=config.log_dir,
            entity=config.wandb_entity,
        )
        self._logger = colorlog.getLogger("file/console")

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

        self.encoder = enc = Encoder(cf.g_dim, cf.channels, cf.multiview).to(
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

        # initialize weights
        for model in self.all_models:
            model.apply(init_weights)

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

    def _train_step(self, data):
        """Forward and Backward pass of models"""
        cf = self._config
        for model in self.all_models:
            model.zero_grad()

        # initialize the recurrent states
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        if cf.stoch:
            self.posterior.hidden = self.posterior.init_hidden()
            self.prior.hidden = self.prior.init_hidden()

        mse = 0
        kld = 0
        x, robot, ac = data

        for i in range(1, cf.n_past + cf.n_future):
            h = self.encoder(x[i - 1])
            r = self.robot_enc(robot[i - 1])
            a = self.action_enc(ac[i - 1])
            h_target = self.encoder(x[i])[0]
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
            mse += mse_criterion(x_pred, x[i])
            if cf.stoch:
                kld += kl_criterion(mu, logvar, mu_p, logvar_p, cf.batch_size)
        loss = mse + kld * cf.beta
        loss.backward()

        self.frame_predictor_optimizer.step()
        if cf.stoch:
            self.posterior_optimizer.step()
            self.prior_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.action_encoder_optimizer.step()
        self.robot_encoder_optimizer.step()
        if cf.stoch:
            return (
                mse.data.cpu().numpy() / (cf.n_past + cf.n_future),
                kld.data.cpu().numpy() / (cf.n_future + cf.n_past),
            )
        else:
            return (mse.data.cpu().numpy() / (cf.n_past + cf.n_future), 0)

    def _eval_epoch(self):
        epoch_mse = []
        epoch_kld = []
        for sequence in self.test_loader:
            # transpose from (B, L, C, W, H) to (L, B, C, W, H)
            frames, robots, actions = sequence
            frames = frames.transpose_(1, 0).to(self._device)
            robots = robots.transpose_(1, 0).to(self._device)
            actions = actions.transpose_(1, 0).to(self._device)
            data = (frames, robots, actions)
            mse, kld = self._eval_step(data)
            epoch_mse.append(mse)
            epoch_kld.append(kld)

        avg_mse = np.mean(epoch_mse)
        avg_kld = np.mean(epoch_kld)
        self._logger.info(f"Eval avg mse: {avg_mse}, avg kld: {avg_kld}")
        eval_stats = {
            "eval/mse": avg_mse,
            "eval/epoch_mse": np.sum(epoch_mse),
            "eval/kld": avg_kld,
            "eval/epoch_kld": np.sum(epoch_kld),
        }
        wandb.log(eval_stats, step=self._step)

    @torch.no_grad()
    def _eval_step(self, data):
        cf = self._config
        # initialize the recurrent states
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        if cf.stoch:
            self.posterior.hidden = self.posterior.init_hidden()
            self.prior.hidden = self.prior.init_hidden()

        mse = 0
        kld = 0
        x, robot, ac = data
        for i in range(1, cf.n_past + cf.n_future):
            h = self.encoder(x[i - 1])
            r = self.robot_enc(robot[i - 1])
            a = self.action_enc(ac[i - 1])
            h_target = self.encoder(x[i])[0]
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
            mse += mse_criterion(x_pred, x[i])
            if cf.stoch:
                kld += kl_criterion(mu, logvar, mu_p, logvar_p, cf.batch_size)
        if cf.stoch:
            return (
                mse.data.cpu().numpy() / (cf.n_past + cf.n_future),
                kld.data.cpu().numpy() / (cf.n_future + cf.n_past),
            )
        else:
            return (mse.data.cpu().numpy() / (cf.n_past + cf.n_future), 0)

    def train(self):
        """Training, Evaluation, Checkpointing loop"""
        cf = self._config
        # load models and dataset
        self._step = self._load_checkpoint()
        self._setup_data()

        # start training
        progress = tqdm(initial=self._step, total=cf.niter * cf.epoch_size)
        for epoch in range(cf.niter):
            for model in self.all_models:
                model.train()
            epoch_kld = epoch_mse = 0
            for _ in range(cf.epoch_size):
                data = next(self.training_batch_generator)
                mse, kld = self._train_step(data)
                epoch_mse += mse
                epoch_kld += kld
                self._step += 1
                wandb.log({"mse": mse, "kld": kld}, step=self._step)
                progress.update()

            # log epoch statistics
            wandb.log({"epoch/mse": epoch_mse, "epoch/kld": epoch_kld}, step=self._step)
            self._logger.info(f"Epoch {epoch}, mse: {epoch_mse}, kld: {epoch_kld}")
            # checkpoint
            if epoch % cf.checkpoint_interval == 0 and epoch > 0:
                self._logger.info(f"Saving checkpoint {epoch}")
                self._save_checkpoint()

            # plot and evaluate on test set
            self.frame_predictor.eval()
            if cf.stoch:
                self.posterior.eval()
                self.prior.eval()
            # self.encoder.eval()
            # self.decoder.eval()
            self._eval_epoch()
            test_data = next(self.testing_batch_generator)
            self.plot(test_data, epoch)
            self.plot_rec(test_data, epoch)

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
                if max_step > num:
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
            ckpt = torch.load(ckpt_path)
            load_models(ckpt)
            step = ckpt["step"]
            return step

    def _setup_data(self):
        """
        Setup the dataset and dataloaders
        """
        train_loader, self.test_loader = create_loaders(config)
        self.training_batch_generator = get_batch(train_loader, self._device)
        self.testing_batch_generator = get_batch(self.test_loader, self._device)

    @torch.no_grad()
    def plot(self, data, epoch):
        """
        Plot the generation with learned prior
        """
        cf = self._config
        x, robot, ac = data

        nsample = 1
        gen_seq = [[] for i in range(nsample)]
        gt_seq = [x[i] for i in range(len(x))]
        for s in range(nsample):
            self.frame_predictor.hidden = self.frame_predictor.init_hidden()
            if self._config.stoch:
                self.posterior.hidden = self.posterior.init_hidden()
                self.prior.hidden = self.prior.init_hidden()
            # first frame of all videos
            gen_seq[s].append(x[0])
            x_in = x[0]
            for i in range(1, cf.n_eval):
                h = self.encoder(x_in)
                r = self.robot_enc(robot[i - 1])
                a = self.action_enc(ac[i - 1])
                if (i == 1 and cf.n_past == 1) or cf.last_frame_skip or i < cf.n_past:
                    h, skip = h
                else:
                    h, _ = h
                h = h.detach()
                if i < cf.n_past:
                    r_target = self.robot_enc(robot[i])
                    h_target = self.encoder(x[i])
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
                    gen_seq[s].append(x_in)

        to_plot = []
        gifs = [[] for t in range(cf.n_eval)]
        nrow = min(cf.batch_size, 10)
        for i in range(nrow):
            # ground truth sequence
            row = []
            for t in range(cf.n_eval):
                row.append(gt_seq[t][i])
            to_plot.append(row)

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

        fname = os.path.join(cf.plot_dir, f"sample_{epoch}.png")
        save_tensors_image(fname, to_plot)

        fname = os.path.join(cf.plot_dir, f"sample_{epoch}.gif")
        save_gif(fname, gifs)

    @torch.no_grad()
    def plot_rec(self, data, epoch):
        """
        Plot the 1 step reconstruction with posterior instead of learned prior
        """
        cf = self._config
        x, robot, ac = data
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        if cf.stoch:
            self.posterior.hidden = self.posterior.init_hidden()
        gen_seq = []
        gen_seq.append(x[0])
        x_in = x[0]
        for i in range(1, cf.n_past + cf.n_future):
            h = self.encoder(x[i - 1])
            r = self.robot_enc(robot[i - 1])
            a = self.action_enc(ac[i - 1])
            h_target = self.encoder(x[i])[0]
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
                gen_seq.append(x_pred)

        to_plot = []
        nrow = min(cf.batch_size, 10)
        for i in range(nrow):
            row = []
            for t in range(cf.n_past + cf.n_future):
                row.append(gen_seq[t][i])
            to_plot.append(row)
        fname = os.path.join(cf.plot_dir, f"rec_{epoch}.png")
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
    from src.config import argparser

    config, _ = argparser()
    make_log_folder(config)
    trainer = PredictionTrainer(config)
    trainer.train()
