from src.dataset.robonet.robonet_dataset import process_batch, get_batch
from src.prediction.trainer import (
    PredictionTrainer,
    make_log_folder,
)
# from src.dataset.locobot.locobot_table_dataloaders import create_locobot_modified_loader
# from src.dataset.locobot.locobot_singleview_dataloader import create_locobot_modified_loader
from src.dataset.franka.franka_dataloader import create_transfer_loader
import ipdb
import numpy as np
from time import time
from tqdm import tqdm


def compute_metrics(cf):
    trainer = PredictionTrainer(cf)
    trainer._step = trainer._load_checkpoint(cf.dynamics_model_ckpt)
    trainer.model.eval()

    test_loader = create_transfer_loader(cf)
    info = trainer._compute_epoch_metrics(test_loader, "test")
    print("PSNR:", info["test/autoreg_psnr"])
    print("SSIM:", info["test/autoreg_ssim"])
    testing_batch_generator = get_batch(test_loader, trainer._device)
    for i in range(1):
       test_data = next(testing_batch_generator)
       trainer.plot(test_data, 0, "test", instance=i)


if __name__ == "__main__":
    """
    Evaluate the checkpoints on locobot modified.

     python -m src.prediction.evaluation.evaluate_checkpoint --jobname 0shotlb_vanilla_modified --wandb False --data_root /home/pallab/locobot_ws/src/eef_control/data --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment finetune_locobot --preprocess_action raw --train_val_split 0.95 --model_use_robot_state False --model_use_mask False --model_use_future_mask False --model_use_future_robot_state False --random_snippet True --lstm_group_norm True --dynamics_model_ckpt checkpoints/vanilla_ckpt_10200.pt --robot_joint_dim 5

     python -m src.prediction.evaluation.evaluate_checkpoint --jobname 0shotlb_roboaware_modified --wandb False --data_root /home/pallab/locobot_ws/src/eef_control/data --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss dontcare_l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment finetune_locobot --preprocess_action raw --train_val_split 0.95 --model_use_robot_state True --model_use_mask True --model_use_future_mask True --model_use_future_robot_state True --random_snippet True --lstm_group_norm True --dynamics_model_ckpt checkpoints/roboaware_ckpt_10200.pt --robot_joint_dim 5

    Evaluate the checkpoints on franka
    # vanilla
    python -m src.prediction.evaluation.evaluate_checkpoint --jobname 0shotlb_vanilla_franka --wandb False --data_root /home/pallab/locobot_ws/src/eef_control/data --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment eval_franka --preprocess_action raw --train_val_split 0.95 --model_use_robot_state False --model_use_mask False --model_use_future_mask False --model_use_future_robot_state False --random_snippet True --lstm_group_norm True --dynamics_model_ckpt checkpoints/vanilla_ckpt_10200.pt --robot_joint_dim 7 --video_length 6
    
    # vanilla + state
    python -m src.prediction.evaluation.evaluate_checkpoint --jobname 0shotlb_vanillastate_franka --wandb False --data_root /home/pallab/locobot_ws/src/eef_control/data --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment eval_franka --preprocess_action raw --train_val_split 0.95 --model_use_robot_state True --model_use_mask False --model_use_future_mask False --model_use_future_robot_state False --random_snippet True --lstm_group_norm True --dynamics_model_ckpt checkpoints/vanillastate_ckpt_10200.pt --robot_joint_dim 7 --video_length 6

    # vanilla + state + future state/mask
    python -m src.prediction.evaluation.evaluate_checkpoint --jobname 0shotlb_vanillastatefuturestatemask_franka --wandb False --data_root /home/pallab/locobot_ws/src/eef_control/data --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment eval_franka --preprocess_action raw --train_val_split 0.95 --model_use_robot_state True --model_use_mask True --model_use_future_mask True --model_use_future_robot_state True --random_snippet True --lstm_group_norm True --dynamics_model_ckpt checkpoints/vanillastatefuturestatemask_ckpt_10200.pt --robot_joint_dim 7 --video_length 6

    # vanilla + state + future state/mask + black robot
    python -m src.prediction.evaluation.evaluate_checkpoint --jobname 0shotlb_vanillastatefuturestatemaskblack_franka --wandb False --data_root /home/pallab/locobot_ws/src/eef_control/data --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment eval_franka --preprocess_action raw --train_val_split 0.95 --model_use_robot_state True --model_use_mask True --model_use_future_mask True --model_use_future_robot_state True --random_snippet True --lstm_group_norm True --dynamics_model_ckpt checkpoints/vanillastatefuturestatemaskblack_ckpt_10200.pt --robot_joint_dim 7 --video_length 6 --black_robot True
    
    # roboaware
    python -m src.prediction.evaluation.evaluate_checkpoint --jobname 0shotlb_roboaware_franka --wandb False --data_root /home/pallab/locobot_ws/src/eef_control/data --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss dontcare_l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment eval_franka --preprocess_action raw --train_val_split 0.95 --model_use_robot_state True --model_use_mask True --model_use_future_mask True --model_use_future_robot_state True --random_snippet True --lstm_group_norm True --dynamics_model_ckpt checkpoints/roboaware_ckpt_10200.pt --robot_joint_dim 7 --video_length 6
    """
    from src.config import create_parser

    parser = create_parser()
    config, unparsed = parser.parse_known_args()
    config.log_dir = "ckpt_eval"
    make_log_folder(config)
    compute_metrics(config)
