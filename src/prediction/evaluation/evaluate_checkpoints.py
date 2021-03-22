import os
from src.prediction.multirobot_trainer import (
    MultiRobotPredictionTrainer,
    make_log_folder,
)
import wandb
import ipdb



def evaluate_checkpoints(config, checkpoint_dir):
    # get all checkpoints
    from glob import glob

    files = glob(os.path.join(checkpoint_dir, "*.pt"))
    files.sort()
    if len(files) == 0:
        return None, None
    # assume filename is ckpt_X.pt
    files_sorted = []
    for f in files:
        name = f.split(".")[0]
        num = int(name.rsplit("_", 1)[-1])
        path = f
        files_sorted.append((num, path))
    files_sorted.sort(key=lambda x: x[0])

    trainer = MultiRobotPredictionTrainer(config)
    trainer._setup_data()
    for ckpt_num, ckpt_path in files_sorted:
        print("evaluating", ckpt_num)
        trainer._load_checkpoint(ckpt_path)
        trainer.model.eval()
        test_info = trainer._compute_epoch_metrics(trainer.test_loader, "test")
        wandb.log(test_info, step=ckpt_num)


        if config.training_regime in ["singlerobot", "train_sawyer_multiview"]:
            transfer_info = trainer._compute_epoch_metrics(
                trainer.transfer_loader, "transfer"
            )
            wandb.log(transfer_info, step=ckpt_num)


if __name__ == "__main__":
    """
    Evaluate the checkpoints.
    job_name: folder name for evaluation info
    """
    from src.config import create_parser
    from torch.multiprocessing import set_start_method

    set_start_method("spawn")

    parser = create_parser()
    parser.add_argument("--ckpt_dir", type=str)
    config, unparsed = parser.parse_known_args()
    config.log_dir = "ckpt_eval"
    make_log_folder(config)
    evaluate_checkpoints(config, config.ckpt_dir)
