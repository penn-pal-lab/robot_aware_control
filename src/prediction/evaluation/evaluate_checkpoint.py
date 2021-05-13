from src.dataset.robonet.robonet_dataset import process_batch, get_batch
from src.prediction.trainer import (
    PredictionTrainer,
    make_log_folder,
)
from src.dataset.locobot.locobot_table_dataloaders import create_locobot_modified_loader
import ipdb
import numpy as np
from time import time
from tqdm import tqdm


def compute_metrics(cf):
    trainer = PredictionTrainer(cf)
    trainer._step = trainer._load_checkpoint(cf.dynamics_model_ckpt)
    trainer.model.eval()

    test_loader = create_locobot_modified_loader(cf)
    info = trainer._compute_epoch_metrics(test_loader, "test")
    print("PSNR:", info["test/autoreg_psnr"])
    print("SSIM:", info["test/autoreg_ssim"])
    testing_batch_generator = get_batch(test_loader, trainer._device)
    for i in range(10):
       test_data = next(testing_batch_generator)
       trainer.plot(test_data, 0, "test", instance=i)


if __name__ == "__main__":
    """
    Evaluate the checkpoints.

    python -m src.prediction.evaluation.evaluate_checkpoint --jobname lbtable_roboaware_rawstateac_tile_norm --wandb False --data_root /scratch/edward/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss dontcare_l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment train_locobot_table --preprocess_action raw --train_val_split 0.95 --model_use_robot_state True --model_use_mask True --model_use_future_mask True --model_use_future_robot_state True --random_snippet True --lstm_group_norm True --dynamics_model_ckpt /scratch/edward/roboaware/logs/lbtable_roboaware_rawstateac_tile_norm/ckpt_136500.pt

    python -m src.prediction.evaluation.evaluate_checkpoint --jobname lbtable_vanilla_rawac_tile_norm --wandb False --data_root /scratch/edward/Robonet --batch_size 16 --n_future 5 --n_past 1 --n_eval 6 --g_dim 256 --z_dim 64 --model svg --niter 100 --epoch_size 300 --eval_interval 15 --checkpoint_interval 5 --reconstruction_loss l1 --last_frame_skip True --scheduled_sampling True --action_dim 5 --robot_dim 5 --data_threads 4 --lr 0.0001 --experiment train_locobot_table --preprocess_action raw --train_val_split 0.95 --model_use_robot_state False --model_use_mask False --model_use_future_mask False --model_use_future_robot_state False --random_snippet True --lstm_group_norm True --dynamics_model_ckpt /scratch/edward/roboaware/logs/lbtable_vanilla_rawac_tile_norm/ckpt_136500.pt
    """
    from src.config import create_parser

    parser = create_parser()
    config, unparsed = parser.parse_known_args()
    config.log_dir = "ckpt_eval"
    make_log_folder(config)
    compute_metrics(config)
