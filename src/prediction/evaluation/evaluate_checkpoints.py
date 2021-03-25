from src.dataset.multirobot_dataset import process_batch
from src.prediction.multirobot_trainer import (
    MultiRobotPredictionTrainer,
    make_log_folder,
)
import ipdb
import tensorflow.compat.v1 as tf
import src.prediction.evaluation.frechet_video_distance.frechet_video_distance as fvd
import numpy as np
from time import time
from tqdm import tqdm


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
    config = tf.ConfigProto(device_count = {'GPU': 0})
    config.gpu_options.allow_growth = True
    fvd_avg = 0
    num_batches = len(test_loader)

    trainer.model.eval()

    for data in tqdm(test_loader):
        start = time()
        data = process_batch(data, trainer._device)
        info = trainer.predict_video(data)

        # B,T,H,W,C size arrays
        true_imgs = np.concatenate(info["true_imgs"], 0)
        gen_imgs = np.concatenate(info["gen_imgs"], 0)
        batch_size = true_imgs.shape[0]
        print(f"video gen done {time() - start:.2f}")
        start = time()
        with tf.Graph().as_default():
            true_imgs_tensor = tf.convert_to_tensor(true_imgs)
            gen_imgs_tensor = tf.convert_to_tensor(gen_imgs)
            result = fvd.calculate_fvd(
                fvd.create_id3_embedding(fvd.preprocess(true_imgs_tensor, (224, 224)), batch_size),
                fvd.create_id3_embedding(fvd.preprocess(gen_imgs_tensor, (224, 224)), batch_size),
            )

            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                fvd_val = sess.run(result)
                fvd_avg += fvd_val / num_batches
                print("FVD is: %.2f." % fvd_val)
        print(f"fvd done {time() - start:.2f} sec")
    print("Final Avg FVD:", fvd_avg)


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
    compute_fid(config)
