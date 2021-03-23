from src.dataset.multirobot_dataset import process_batch
from src.prediction.multirobot_trainer import (
    MultiRobotPredictionTrainer,
    make_log_folder,
)
import ipdb
import tensorflow.compat.v1 as tf
import src.prediction.evaluation.frechet_video_distance.frechet_video_distance as fvd


def compute_fid(cf):
    trainer = MultiRobotPredictionTrainer(cf)
    trainer._load_checkpoint(cf.dynamics_model_ckpt)
    trainer.model.eval()
    from src.dataset.sawyer_multiview_dataloaders import (
        create_finetune_loaders as create_loaders,
    )

    # assuming 400 test videos, that's around 12000 images of prediction
    # so 24k images total to fit on GPU for FVD.
    train_loader, test_loader, comp_loader = create_loaders(cf)
    del train_loader
    del comp_loader
    # generate predictions
    for data in test_loader:
        data = process_batch(data, trainer._device)
        info = trainer.predict_video(data)
        # B,H,W,C size arrays
        true_imgs = info["true_imgs"]
        gen_imgs = info["gen_imgs"]
        with tf.Graph().as_default():
            true_imgs_tensor = tf.convert_to_tensor(true_imgs)
            gen_imgs = tf.convert_to_tensor(gen_imgs)
            result = fvd.calculate_fvd(
                fvd.create_id3_embedding(fvd.preprocess(true_imgs_tensor, (224, 224))),
                fvd.create_id3_embedding(fvd.preprocess(gen_imgs, (224, 224))),
            )
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                print("FVD is: %.2f." % sess.run(result))

        break


if __name__ == "__main__":
    """
    Evaluate the checkpoints.
    job_name: folder name for evaluation info
    """
    from src.config import create_parser

    parser = create_parser()
    parser.add_argument("--ckpt_dir", type=str)
    config, unparsed = parser.parse_known_args()
    config.log_dir = "ckpt_eval"
    make_log_folder(config)
