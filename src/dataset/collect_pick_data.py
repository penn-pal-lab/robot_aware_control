from functools import partial
import os
from src.utils.mujoco import init_mjrender_device

import h5py
import imageio
import numpy as np
from src.config import argparser
from src.env.robotics.locobot_pick_env import LocobotPickEnv
from tqdm import tqdm


def generate_demos(rank, config, record, num_trajectories):
    """
    rank: idx of the worker
    config: configuration
    record: whether to record the gif or not
    num_trajectories: number of demos to generate
    ep_len: max length of the demo. only used by some behaviors.
    noise: action noise in behavior. only used by some behaviors
    """
    config.seed = rank
    env = LocobotPickEnv(config)
    len_stats = []
    it = range(num_trajectories)
    if rank == 0:
        it = tqdm(it)
    for i in it:
        record = rank == 0 and record
        name = f"pick_{rank}_{i}.hdf5"
        path = os.path.join(config.demo_dir, name)
        history = env.generate_demo()
        record_path = f"videos/pick_{config.seed}_{i}.gif"
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

        # make 3d action 5d for video prediction
        actions = [np.array([ac[0], ac[1], ac[2], 0, ac[3]]) for ac in history["ac"]]
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
    stats_path = os.path.join(config.demo_dir, f"stats_pick_{config.seed}.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)


def create_demo_dataset(config, num_demo, num_workers, record):
    """
    Collect all demonstrations and save into demo_dir
    You can use multiple workers if generating 1000s of demonstrations
    """
    from multiprocessing import Process

    os.makedirs(config.demo_dir, exist_ok=True)
    if num_workers == 1:
        generate_demos(0, config, record, num_demo)
    else:
        ps = []
        for i in range(num_workers):
            p = Process(
                target=generate_demos,
                args=(i, config,record, num_demo),
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
    num_workers = 1
    num_demos = 10 // num_workers
    record = False
    MODIFIED = False

    config, _ = argparser()
    config.gpu = 0
    init_mjrender_device(config)

    config.modified = MODIFIED
    config.demo_dir = f"/scratch/edward/Robonet/locobot_pick{'_fetch' if MODIFIED else ''}_views/c0"
    # config.demo_dir = f"/home/pallab/locobot_ws/src/roboaware/demos"
    create_demo_dataset(config, num_demos, num_workers, record)


if __name__ == "__main__":
    """
    Use this to collect demonstrations for svg / demo cem experiments
    """
    collect_svg_data()
