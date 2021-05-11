from src.dataset.robonet.robonet_dataset import process_batch, get_batch
from src.prediction.trainer import (
    PredictionTrainer,
    make_log_folder,
)
from src.dataset.robonet.robonet_dataloaders import create_movement_loader
from src.dataset.locobot.locobot_table_dataloaders import create_locobot_modified_loader
import ipdb
import numpy as np
from time import time
from tqdm import tqdm


def compute_metrics(cf):
    trainer = PredictionTrainer(cf)
    trainer._step = trainer._load_checkpoint(cf.dynamics_model_ckpt)
    trainer.model.eval()

    # test_loader = create_movement_loader(cf)
    test_loader = create_movement_loader
    info = trainer._compute_epoch_metrics(test_loader, "test")
    print("PSNR:", info["test/autoreg_psnr"])
    print("SSIM:", info["test/autoreg_ssim"])
    #testing_batch_generator = get_batch(test_loader, trainer._device)
    #for i in range(10):
    #    test_data = next(testing_batch_generator)
    #    trainer.plot(test_data, 0, "test", instance=i)


if __name__ == "__main__":
    """
    Evaluate the checkpoints.
    """
    from src.config import create_parser

    parser = create_parser()
    config, unparsed = parser.parse_known_args()
    config.log_dir = "ckpt_eval"
    make_log_folder(config)
    compute_metrics(config)
