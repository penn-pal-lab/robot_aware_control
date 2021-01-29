import ipdb
import numpy as np
import tensorflow as tf
import imageio
import argparse
import os

from robonet.robonet.datasets import load_metadata
from robonet.robonet.datasets.robonet_dataset import RoboNetDataset
from robonet.robonet.datasets.util.hdf5_loader import (
    load_camera_imgs, load_data,
    load_qpos,
    default_loader_hparams,
    load_data_customized,
)
import h5py

# when this number if large enough, we collect all corresponding data
NUM_VISUAL_PER_VIEW = 1000


def collect_same_viewpoint(robot, directory):
    hparams = tf.contrib.training.HParams(**default_loader_hparams())

    meta_data = load_metadata(directory)

    exp_same_view = {}

    for f in os.listdir(directory):
        if robot in f:
            path = directory + f
            print(path)
            _, _, _, _, _, viewpoint = load_data_customized(
                path, meta_data.get_file_metadata(path), hparams
            )
            if viewpoint not in exp_same_view:
                exp_same_view[viewpoint] = [path]
            else:
                exp_same_view[viewpoint].append(path)
    return exp_same_view

def generate_hdf5_video(directory, hdf5_path, hparams, video_path):
    meta_data = load_metadata(directory)
    file_metadata = meta_data.get_file_metadata(hdf5_path)
    with h5py.File(hdf5_path) as hf:
        start_time, n_states = 0, min([file_metadata['state_T'], file_metadata['img_T'], file_metadata['action_T'] + 1])
        assert n_states > 1, "must be more than one state in loaded tensor!"
        assert all([0 <= i < file_metadata['ncam'] for i in hparams.cams_to_load]), "cams_to_load out of bounds!"
        images, selected_cams = [], []
        for cam_index in hparams.cams_to_load:
            images.append(load_camera_imgs(cam_index, hf, file_metadata, hparams.img_size, start_time, n_states)[None])
            selected_cams.append(cam_index)
        imgs = np.swapaxes(np.concatenate(images, 0), 0, 1)
        writer = imageio.get_writer(video_path)
        for t in range(imgs.shape[0]):
            gif_img = np.concatenate(imgs[t, :], axis=1)
            writer.append_data(gif_img)
        writer.close()

def generate_calibration_data(dataset_dir, target_dir, hdf5_list, hparams):
    """
    Given a list of trajectories, load them and save them into
    the directory for calibration pipeline.
    """
    os.makedirs(target_dir, exist_ok=True)
    meta_data = load_metadata(dataset_dir)
    for hdf5_path in hdf5_list:
        file_metadata = meta_data.get_file_metadata(hdf5_path)
        exp_name = hdf5_path.split("/")[-1][:-5]
        imgs, states, qposes, _, _, _ = load_data_customized(
            hdf5_path, file_metadata, hparams
        )
        print("saving experiment:", exp_name)
        np.save(target_dir + "/states_" + exp_name, states)
        np.save(target_dir + "/qposes_" + exp_name, qposes)

        writer = imageio.get_writer(target_dir + "/" + exp_name + ".gif")
        for t in range(imgs.shape[0]):
            gif_img = imgs[t, 0]
            writer.append_data(gif_img)
        writer.close()

def generate_video(hparams):
    robonet_root = "/media/ed/hdd/Datasets/Robonet/hdf5/"
    traj = "/media/ed/hdd/Datasets/Robonet/hdf5/penn_kuka_traj0.hdf5"
    video_path = "kuka1.gif"
    generate_hdf5_video(robonet_root, traj, hparams, video_path)

def generate_viewpoint_calibration_data(args, hparams):
    meta_data = load_metadata(args.directory)

    exp_same_view = collect_same_viewpoint(args.robot, args.directory)
    print(len(exp_same_view))

    os.makedirs(args.robot, exist_ok=True)
    for vp in exp_same_view:
        target_folder = args.robot + "/" + vp
        os.makedirs(target_folder, exist_ok=True)
        visuals = min(NUM_VISUAL_PER_VIEW, len(exp_same_view[vp]))
        for i in range(visuals):
            f = exp_same_view[vp][i]
            exp_name = f.split("/")[-1][:-5]
            imgs, states, qposes, ws_min, ws_max, viewpoint = load_data_customized(
                f, meta_data.get_file_metadata(f), hparams
            )
            print("saving experiment:", exp_name)
            np.save(target_folder + "/states_" + exp_name, states)
            np.save(target_folder + "/qposes_" + exp_name, qposes)

            writer = imageio.get_writer(target_folder + "/" + exp_name + ".gif")
            for t in range(imgs.shape[0]):
                gif_img = np.concatenate([imgs[t, 0], imgs[t, 1], imgs[t, 2]], axis=1)
                writer.append_data(gif_img)
            writer.close()
def generate_kuka_calibration(args, hparams):
    robonet_root = "/media/ed/hdd/Datasets/Robonet/hdf5/"
    kuka_paths = ["penn_kuka_traj0", "penn_kuka_traj1", "penn_kuka_traj10"]
    hdf5_paths = [os.path.join(robonet_root, f"{k}.hdf5") for k in kuka_paths]
    dataset_dir = args.directory
    target_dir = "kuka_calibration"
    generate_calibration_data(dataset_dir, target_dir, hdf5_paths, hparams)

def generate_widowx_calibration(args, hparams):
    robonet_root = "/media/ed/hdd/Datasets/Robonet/hdf5/"
    kuka_paths = ["berkeley_widowx_traj1000", "berkeley_widowx_traj1170", "berkeley_widowx_traj1300"]
    hdf5_paths = [os.path.join(robonet_root, f"{k}.hdf5") for k in kuka_paths]
    dataset_dir = args.directory
    target_dir = "widowx_calibration"
    generate_calibration_data(dataset_dir, target_dir, hdf5_paths, hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect data corresponding to specific robot, and organize them by viewpoints"
    )
    parser.add_argument("--directory", type=str, help="path to dataset folder")
    parser.add_argument("--robot", type=str, help="robot")
    args = parser.parse_args()

    hparams = tf.contrib.training.HParams(**default_loader_hparams())
    hparams.img_size = [240, 320]
    hparams.cams_to_load = [0]

    # generate_video(hparams)
    # generate_viewpoint_calibration_data(args, hparams)
    # generate_kuka_calibration(args, hparams)
    generate_widowx_calibration(args, hparams)
