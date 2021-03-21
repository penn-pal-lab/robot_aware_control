import logging
import os
from collections import defaultdict
from functools import partial
from src.env.robotics.masks.widowx_mask_env import WidowXMaskEnv
from src.env.robotics.masks.sawyer_mask_env import SawyerMaskEnv
from src.env.robotics.masks.baxter_mask_env import BaxterMaskEnv


from src.dataset.joint_pos_dataset import get_batch, process_batch
from src.prediction.models.dynamics import GripperStatePredictor, JointPosPredictor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from warnings import simplefilter  # disable tensorflow warnings

simplefilter(action="ignore", category=FutureWarning)

from math import floor

import colorlog
import imageio
import ipdb
import numpy as np
import torch
import wandb

from src.prediction.losses import mse_criterion
from src.utils.plot import save_gif
from torch import optim
from tqdm import tqdm
from src.utils.camera_calibration import world_to_camera_dict
import torchvision.transforms as tf



class RobotPredictionTrainer(object):
    """
    Qpos, EEF Prediction with Sawyer / Baxter
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
            project="robot-prediction",
            config=config,
            dir=config.log_dir,
            entity=config.wandb_entity,
            group=config.wandb_group,
            job_type=config.wandb_job_type,
            config_exclude_keys=["device"],
        )
        self._img_augmentation = config.img_augmentation
        self._plot_rng = np.random.RandomState(self._config.seed)
        self._img_transform = tf.Compose(
            [tf.ToTensor(), tf.CenterCrop(config.image_width)]
        )

        if config.training_regime == "singlerobot":
            self._robot_sim = SawyerMaskEnv()
        elif config.training_regime == "finetune":
            self._robot_sim = BaxterMaskEnv()
        elif config.training_regime == "finetune_widowx":
            self._robot_sim = WidowXMaskEnv()

    def _init_models(self, cf):
        """Initialize models and optimizers
        When adding a new model, make sure to:
        - Call to(device)
        - Add optimizer
        - Add optimizer step() call
        - Update save and load ckpt code
        """

        if cf.optimizer == "adam":
            optimizer = partial(optim.Adam, lr=cf.lr, betas=(cf.beta1, 0.999))
        elif cf.optimizer == "rmsprop":
            optimizer = optim.RMSprop
        elif cf.optimizer == "sgd":
            optimizer = optim.SGD
        else:
            raise ValueError("Unknown optimizer: %s" % cf.optimizer)

        self.joint_model = JointPosPredictor(cf).to(self._device)
        self.gripper_model = GripperStatePredictor(cf).to(self._device)
        params = list(self.joint_model.parameters()) + list(self.gripper_model.parameters())
        self.optimizer = optimizer(params)

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

    def _joint_loss(self, prediction, target):
        # try weighted variant
        # K = self._config.robot_joint_dim
        # diff = target - prediction
        # # give more weight to earlier joints
        # weights = torch.from_numpy(np.array([1/K * i for i in range(1, K+1)][::-1])).to(self._device)
        # mse = torch.mean(diff ** 2, (0)) # get per dim mse
        # weighted_mse = torch.sum(weights * mse)
        # return weighted_mse
        return mse_criterion(prediction, target)


    def _gripper_loss(self, prediction, target):
        return mse_criterion(prediction, target)

    def _train_video(self, data):
        """Train the model over the video data

        Slices video up into K length sequences for training.
        Args:
            data (dict): Video data
        """
        T = data["qpos"].shape[0]
        window = self._config.n_past + self._config.n_future
        all_losses = defaultdict(float)
        for i in range(floor(T / window)):
            s = i * window
            e = (i + 1) * window
            batch_data = {
                "states": data["states"][s:e],
                "actions": data["actions"][s : e - 1],
                "qpos": data["qpos"][s:e],
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
        joint_loss = gripper_loss = 0
        states = data["states"]
        ac = data["actions"]
        qpos = data["qpos"]
        self.joint_model.zero_grad()
        self.gripper_model.zero_grad()

        for i in range(1, cf.n_past + cf.n_future):
            # let j be i - 1, or previous timestep
            r_j, a_j = states[i - 1], ac[i - 1]
            q_j, q_i = qpos[i - 1], qpos[i]
            a_j = ac[i - 1]
            q_pred = self.joint_model(q_j,  a_j) + q_j
            q_loss = self._joint_loss(q_pred, q_i)

            joint_loss += q_loss
            losses[f"joint_loss"] += q_loss.cpu().item()

            joint_mse = mse_criterion(q_pred, q_i)
            losses[f"joint_mse"] += joint_mse.cpu().item()

            # qpos-> eef pos loss
            # all_pred_eef = []
            # all_true_eef = []
            # q_pred_d = q_pred.clone().detach()
            # q_j_d = q_j.clone().detach()
            # for k in range(len(q_pred)):
            #     q_p = q_pred_d[k]
            #     true_q = q_j_d[k]
            #     pred_eef = self._get_gripper_pos(q_p.cpu().numpy())
            #     true_eef = self._get_gripper_pos(true_q.cpu().numpy())
            #     all_pred_eef.append(pred_eef)
            #     all_true_eef.append(true_eef)
            # all_pred_eef = torch.from_numpy(np.asarray(all_pred_eef))
            # all_true_eef = torch.from_numpy(np.asarray(all_true_eef))
            # eef_mse = mse_criterion(all_pred_eef, all_true_eef)
            # losses[f"joint_eef_pos_loss"] += eef_mse.cpu().item()

             # gripper loss
            r_i = states[i]
            r_pred = self.gripper_model(r_j, a_j) + r_j
            r_loss = self._gripper_loss(r_pred, r_i)
            gripper_loss += r_loss
            losses[f"gripper_loss"] += r_loss.cpu().item()

        loss = joint_loss + gripper_loss
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
        T = data["qpos"].shape[0]
        window = self._config.n_eval
        all_losses = defaultdict(float)
        for i in range(floor(T / window)):
            s = i * window
            e = (i + 1) * window
            batch_data = {
                "states": data["states"][s:e],
                "actions": data["actions"][s : e - 1],
                "qpos": data["qpos"][s:e],
                "robot": data["robot"],
            }
            losses = self._eval_step(batch_data, autoregressive)
            for k, v in losses.items():
                all_losses[k] += v
        for k, v in all_losses.items():
            all_losses[k] /= floor(T / window)
        return all_losses

    @torch.no_grad()
    def _eval_step(self, data, autoregressive=False):
        """
        Evals over a snippet of video of length n_past + n_future
        autoregressive: use model's outputs as input for next timestep
        """
        # one step evaluation loss
        cf = self._config
        joint_loss = gripper_loss = 0

        states = data["states"]
        ac = data["actions"]
        qpos = data["qpos"]
        losses = defaultdict(float)
        prefix = "autoreg" if autoregressive else "1step"
        for i in range(1, cf.n_eval):
            if autoregressive and i > 1:
                q_j = q_pred.clone().detach()
                r_j = r_pred.clone().detach()
            else:
                q_j = qpos[i - 1]
                r_j = states[i - 1]
            # let j be previous timestep
            a_j =  ac[i - 1]
            q_i = qpos[i]
            q_pred = self.joint_model(q_j, a_j) + q_j
            q_loss = self._joint_loss(q_pred, q_i)
            joint_loss += q_loss
            losses[f"{prefix}_joint_loss"] += q_loss.cpu().item()
            joint_mse = mse_criterion(q_pred, q_i)
            losses[f"{prefix}_joint_mse"] += joint_mse.cpu().item()

            # qpos-> eef pos loss
            all_pred_eef = []
            all_true_eef = []
            for k in range(len(q_pred)):
                q_p = q_pred[k]
                true_q = q_j[k]
                pred_eef = self._get_gripper_pos(q_p.cpu().numpy())
                true_eef = self._get_gripper_pos(true_q.cpu().numpy())
                all_pred_eef.append(pred_eef)
                all_true_eef.append(true_eef)
            all_pred_eef = torch.from_numpy(np.asarray(all_pred_eef))
            all_true_eef = torch.from_numpy(np.asarray(all_true_eef))
            eef_mse = mse_criterion(all_pred_eef, all_true_eef)
            losses[f"{prefix}_joint_eef_pos_loss"] += eef_mse.cpu().item()

            # gripper loss
            r_i = states[i]
            r_pred = self.gripper_model(r_j, a_j) + r_j
            r_loss = self._gripper_loss(r_pred, r_i)
            gripper_loss += r_loss
            losses[f"{prefix}_gripper_loss"] += r_loss.cpu().item()

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
            self.joint_model.train()
            self.gripper_model.train()
            for i in range(cf.epoch_size):
                data = next(self.training_batch_generator)
                info = self._train_video(data)
                if self._scheduled_sampling:
                    info["sample_schedule"] = self._schedule_prob()[0]
                self._step += 1

                if i == cf.epoch_size - 1:
                    self.plot(data, epoch, "train")

                wandb.log({f"train/{k}": v for k, v in info.items()}, step=self._step)
                self.progress.update()

            if epoch % cf.checkpoint_interval == 0 and epoch > 0:
                self._logger.info(f"Saving checkpoint {epoch}")
                self._save_checkpoint()
            if epoch % cf.eval_interval == 0:
                # plot and evaluate on test set
                self.joint_model.eval()
                self.gripper_model.eval()
                info = self._compute_epoch_metrics(self.test_loader, "test")
                wandb.log(info, step=self._step)
                test_data = next(self.testing_batch_generator)
                self.plot(test_data, epoch, "test")


    def _save_checkpoint(self):
        path = os.path.join(self._config.log_dir, f"ckpt_{self._step}.pt")
        data = {
            "joint_model": self.joint_model.state_dict(),
            "gripper_model": self.gripper_model.state_dict(),
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
                self.joint_model.load_state_dict(ckpt["joint_model"])
                self.gripper_model.load_state_dict(ckpt["gripper_model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                step = ckpt["step"]
                return step
        else:
            # load given ckpt path
            print(f"Loading ckpt {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self._device)
            self.joint_model.load_state_dict(ckpt["joint_model"])
            self.gripper_model.load_state_dict(ckpt["gripper_model"])
            if self._config.training_regime in ["finetune", "finetune_sawyer_view", "finetune_widowx"]:
                step = 0
            else:
                step = ckpt["step"]
                self.optimizer.load_state_dict(ckpt["optimizer"])
            return step

    def _setup_data(self):
        """
        Setup the dataset and dataloaders
        """
        if self._config.training_regime == "singlerobot":
            from src.dataset.sawyer_joint_pos_dataloaders import create_joint_pos_loaders as create_loaders
        elif self._config.training_regime == "finetune":
            from src.dataset.baxter_joint_pos_dataloaders import create_finetune_loaders as create_loaders
        elif  self._config.training_regime == "finetune_widowx":
            from src.dataset.widowx_joint_pos_dataloaders import create_loaders
        else:
            raise NotImplementedError(self._config.training_regime)
        self.train_loader, self.test_loader = create_loaders(self._config)
        # for infinite batching
        self.training_batch_generator = get_batch(self.train_loader, self._device)
        self.testing_batch_generator = get_batch(self.test_loader, self._device)

    def _get_gripper_pos(self, qpos):
        """
        Compute forward kinematics to get the gripper position. Used for error metrics.
        """

        if self._config.training_regime == "finetune": # baxter
            joint_references = [self._robot_sim.sim.model.get_joint_qpos_addr(f"left_{x}") for x in self._robot_sim._joints]
            self._robot_sim.sim.data.qpos[joint_references] = qpos
            self._robot_sim.sim.forward()
            return self._robot_sim.sim.data.get_body_xpos("left_gripper").copy()
        elif config.training_regime == "singlerobot": # sawyer
            self._robot_sim.sim.data.qpos[self._robot_sim._joint_references] = qpos
            self._robot_sim.sim.forward()
            return self._robot_sim.sim.data.get_body_xpos("right_hand").copy()
        elif config.training_regime == "finetune_widowx": # sawyer
            self._robot_sim.sim.data.qpos[self._robot_sim._joint_references] = qpos
            self._robot_sim.sim.forward()
            return self._robot_sim.sim.data.get_body_xpos("wrist_2_link").copy()

    @torch.no_grad()
    def plot(self, data, epoch, name, random_start=True):
        """Project masks with learned model. Autoregressive output.
        Args:
            data (DataLoader): dictionary from dataloader
            epoch (int): epoch number
            name (str): name of the dataset
            random_start (bool, optional): Choose a random timestep as the starting frame
        """
        cf = self._config
        robot = data["states"]
        ac = data["actions"]
        mask = data["masks"]
        qpos = data["qpos"]
        # initialize rendering environments for projection
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
                cam_ext = world_to_camera_dict["widowx1"]

            env.set_opencv_camera_pose("main_cam", cam_ext)
            self.renderers[v] = env

        b = min(qpos.shape[1], 10)
        # first frame of all videos
        start = 0
        video_len = cf.n_eval
        if name in ["comparison", "train"]:
            video_len = cf.n_past + cf.n_future
        end = start + video_len
        if random_start:
            offset = qpos.shape[0] - video_len
            start = self._plot_rng.randint(0, offset + 1)
            end = start + video_len
        # truncate batch by time and batch dim
        robot = robot[start:end, :b]
        ac = ac[start : end - 1, :b]
        mask = mask[start:end, :b]
        qpos = qpos[start:end, :b]

        # first get all predicted qpos
        all_q_preds = [qpos[0]]
        for i in range(1, video_len):
            if i > 1:
                q_j = q_pred.clone().detach()
            else:
                q_j = qpos[i - 1]
            # let j be previous timestep
            a_j = robot[i - 1], ac[i - 1]
            a_j = ac[i - 1]
            q_pred = self.joint_model(q_j, a_j) + q_j
            all_q_preds.append(q_pred)

        # project qpos into masks, compare with real masks
        all_masks = []
        all_iou = []
        for idx in range(b):
            vp = data["file_name"][idx] # get vp for each batch sample
            env = self.renderers[vp]
            qpos_list = []
            for i in range(video_len):
                qpos = all_q_preds[i][idx].cpu().numpy()
                qpos_list.append(qpos)
            mask_list = env.generate_masks(qpos_list) # |T| array
            # mask_list = torch.from_numpy(mask_list)
            mask_list = torch.stack([self._img_transform(i) for i in mask_list])
            true_mask_list = mask[:, idx].cpu() # T x |I| tensor
            # compute IoU of the masks
            video_IoU = self._compute_iou(mask_list, true_mask_list).mean().item()
            # generate a difference map
            diff_mask = (true_mask_list.type(torch.bool) ^ mask_list).type(torch.float32)
            comparison_list = torch.stack([true_mask_list,mask_list, diff_mask]) # 3 x T x |I| tensor
            all_iou.append(video_IoU)
            all_masks.append(comparison_list)
        all_masks = torch.stack(all_masks) # B x 3 x T x |I| tensor
        # T x B x 3 x |I| array of the comparison images
        all_masks = all_masks.transpose_(1,2).transpose_(0,1)
        fname = os.path.join(cf.plot_dir, f"{name}_{epoch}.gif")
        save_gif(fname, all_masks)
        wandb.log({f"{name}/gifs": wandb.Video(fname, format="gif")}, step=self._step)
        wandb.log({f"{name}/IoU": np.mean(all_iou)}, step=self._step)

    def _compute_iou(self, prediction, target):
        prediction = prediction.type(torch.bool)
        target = target.type(torch.bool)
        I = prediction & target
        U = prediction | target
        IoU = I.sum((1,2,3)) / U.sum((1,2,3))
        return IoU





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
    trainer = RobotPredictionTrainer(config)
    trainer.train()
