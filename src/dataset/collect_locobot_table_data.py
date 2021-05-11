from functools import partial
import os
from collections import defaultdict
from src.utils.mujoco import init_mjrender_device
from src.utils.plot import putText

import h5py
import imageio
import numpy as np
from src.config import argparser
from src.env.robotics.locobot_table_env import LocobotTableEnv
from tqdm import tqdm


def generate_demos(rank, config, behavior, record, num_trajectories):
    """
    rank: idx of the worker
    config: configuration
    behavior: what the robot will do. see env.generate_demo for types
    record: whether to record the gif or not
    num_trajectories: number of demos to generate
    ep_len: max length of the demo. only used by some behaviors.
    noise: action noise in behavior. only used by some behaviors
    """
    config.seed = rank
    env = LocobotTableEnv(config)
    len_stats = []
    it = range(num_trajectories)
    if rank == 0:
        it = tqdm(it)
    for i in it:
        record = rank == 0 and record
        name = f"{behavior}_{rank}_{i}.hdf5"
        path = os.path.join(config.demo_dir, name)
        history = env.generate_demo(behavior)
        record_path = f"videos/{behavior}_{config.seed}_{i}.gif"
        obs = history["obs"]  # array of observation dictionaries
        len_stats.append(len(obs))
        states = []
        masks = []
        imgs = []
        qpos = []
        for ob in obs:
            imgs.append(ob["observation"])
            masks.append(ob["masks"])
            states.append(ob["states"])
            qpos.append(ob["qpos"])

        actions = history["ac"]
        # now render the object only demonstration
        if record:
            imageio.mimwrite(record_path, imgs)
        with h5py.File(path, "w") as hf:
            create_dataset = partial(hf.create_dataset, compression="gzip")
            create_dataset("actions", data=actions)
            create_dataset("observations", data=imgs)
            create_dataset("states", data=states)
            create_dataset("qpos", data=qpos)
            create_dataset("masks", data=masks)

    # print out stats about the dataset
    stats_str = f"Avg len: {np.mean(len_stats)}\nstd: {np.std(len_stats)}\nmin: {np.min(len_stats)}\nmax: {np.max(len_stats)}\n"
    print(stats_str)
    stats_path = os.path.join(config.demo_dir, f"stats_{behavior}_{config.seed}.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)


def create_demo_dataset(config, num_demo, num_workers, record, behavior):
    """
    Collect all demonstrations and save into demo_dir
    You can use multiple workers if generating 1000s of demonstrations
    """
    from multiprocessing import Process

    os.makedirs(config.demo_dir, exist_ok=True)
    if num_workers == 1:
        generate_demos(0, config, behavior, record, num_demo)
    else:
        ps = []
        for i in range(num_workers):
            p = Process(
                target=generate_demos,
                args=(i, config, behavior, record, num_demo),
            )
            ps.append(p)

        for p in ps:
            p.start()

        for p in ps:
            p.join()


def collect_svg_data():
    """
    Generate video dataset for SVG model training
    """
    num_workers = 5
    num_push = 10000 // num_workers
    record = False

    config, _ = argparser()
    config.gpu = 0
    init_mjrender_device(config)

    config.demo_dir = "/scratch/edward/Robonet/locobot_table_views/c0"
    config.img_dim = 64
    config.camera_ids = [0]
    config.temporal_beta = 0.2  # control random policy's temporal correlation
    config.action_noise = 0.05
    config.demo_length = 30 # actually 31
    create_demo_dataset(config, num_push, num_workers, record, "temporal_random_robot")


if __name__ == "__main__":
    """
    Use this to collect demonstrations for svg / demo cem experiments
    """
    collect_svg_data()
