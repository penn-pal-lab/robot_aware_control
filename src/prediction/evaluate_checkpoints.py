import os
from src.prediction.multirobot_trainer import (
    MultiRobotPredictionTrainer,
    make_log_folder,
)
import wandb



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
    print(files_sorted)

    trainer = MultiRobotPredictionTrainer(config)
    trainer._setup_data()
    for ckpt_path, ckpt_num in files_sorted:
        print("evaluating", ckpt_num)
        trainer._load_checkpoint(ckpt_path)
        trainer.model.eval()
        test_info = trainer._compute_epoch_metrics(trainer.test_loader, "test")
        wandb.log(test_info, step=ckpt_num)
        test_data = next(trainer.testing_batch_generator)
        trainer.plot(test_data, ckpt_num, "test")
        trainer.plot_rec(test_data, ckpt_num, "test")

        if config.training_regime in ["singlerobot", "train_sawyer_multiview"]:
            transfer_info = trainer._compute_epoch_metrics(
                trainer.transfer_loader, "transfer"
            )
            wandb.log(transfer_info, step=ckpt_num)
            transfer_data = next(trainer.transfer_batch_generator)
            trainer.plot(transfer_data, ckpt_num, "transfer")


if __name__ == "__main__":
    """
    Evaluate the checkpoints.
    checkpoint_dir: the path to the checkpoints for evaluation
    job_name: folder name for evaluation info
    """
    from src.config import argparser

    config, _ = argparser()
    checkpoint_dir = "asdf"
    config.log_dir = "ckpt_eval"
    config.job_name = "ckpt_eval_model_name_here"
    make_log_folder(config)
    evaluate_checkpoints(config, checkpoint_dir)
