import numpy as np
import tensorflow as tf
import imageio
import argparse

from robonet.datasets import load_metadata
from robonet.datasets.robonet_dataset import RoboNetDataset
from robonet.datasets.util.hdf5_loader import (
    load_data,
    load_qpos,
    default_loader_hparams,
    load_data_customized,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="tests hdf5 data loader without tensorflow dataset wrapper"
    )
    parser.add_argument("file", type=str, help="path to hdf5 you want to load")
    parser.add_argument(
        "--load_annotations", action="store_true", help="loads annotations if supplied"
    )
    parser.add_argument(
        "--load_steps",
        type=int,
        default=0,
        help="loads <load_steps> steps from the dataset instead of everything",
    )
    args = parser.parse_args()

    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.load_T = args.load_steps
    hparams.img_size = [240, 320]

    assert "hdf5" in args.file
    exp_name = args.file.split("/")[-1][:-5]
    data_folder = "/".join(args.file.split("/")[:-1])
    meta_data = load_metadata(data_folder)

    imgs, states, qposes, ws_min, ws_max, viewpoint = load_data_customized(
        args.file, meta_data.get_file_metadata(args.file), hparams
    )
    print("states", states.shape)
    print("images", imgs.shape)
    print("qposes", qposes.shape)

    print("saving experiment:", exp_name)
    np.save("images/states_" + exp_name, states)
    np.save("images/qposes_" + exp_name, qposes)

    writer = imageio.get_writer("images/" + exp_name + ".gif")
    for t in range(imgs.shape[0]):
        imageio.imwrite("images/" + exp_name + "_" + str(t) + ".png", imgs[t, 0])
        print("state:   ", states[t])
        print("qpos:    ", qposes[t])
        for i in range(1):
            writer.append_data(imgs[t, 0])
    writer.close()
