from functools import partial
import os
from collections import defaultdict
from src.utils.plot import putText

import h5py
import imageio
import numpy as np
from src.config import argparser
from src.env.fetch.clutter_push import ClutterPushEnv
from tqdm import tqdm


def generate_demos(rank, config, behavior, record, num_trajectories, ep_len):
    """
    This generates demos, like random moving or block pushing.

    We first have the robot perform the behavior, save the trajectory, and render it with the robot inpainted, and also render with the robot in scene.
    Next, we replay the trajectory, but move the robot out of scene and move the block without the robot.

    This results in 3 types of video for the dataset.
    1. Video of inpainted robot pushing block
    2. Video of robot pushing block
    3. Video of blocks moving by themselves. This is equivalent to using a perfect inpainting method.

    rank: idx of the worker
    config: configuration
    behavior: what the robot will do. see env.generate_demo for types
    record: whether to record the gif or not
    num_trajectories: number of demos to generate
    ep_len: max length of the demo. only used by some behaviors.
    noise: action noise in behavior. only used by some behaviors
    """
    config.seed = rank
    env = ClutterPushEnv(config)
    len_stats = []
    it = range(num_trajectories)
    all_frames = []
    all_world_coord = []
    if rank == 0:
        it = tqdm(it)
    for i in it:
        # only record first episode for sanity check
        record = rank == 0 and record
        name = f"{behavior}_{rank}_{i}.hdf5"
        path = os.path.join(config.demo_dir, name)

        history = env.generate_demo(behavior)
        obs = history["obs"]  # array of observation dictionaries
        len_stats.append(len(obs))
        frames = []
        # robot = []
        world_coord = []
        for ob in obs:
            world_coord.append(ob["world_coord"].transpose(1, 2, 0))
            frames.append(ob["observation"])
            # robot.append(ob["robot"])

        frames = np.asarray(frames)
        world_coord = np.asarray(world_coord)

        all_world_coord.append(world_coord)
        all_frames.append(frames)
        if record:
            record_path = f"videos/{behavior}_{config.seed}_{i}.gif"
            imageio.mimwrite(record_path, frames)
        # robot = np.asarray(robot)
        # actions = history["ac"]
        # assert len(frames) - 1 == len(actions)
    with h5py.File(path, "w") as hf:
        for i, (frame, world_coord) in tqdm(
            enumerate(zip(all_frames, all_world_coord))
        ):
            hf.create_dataset(f"frame_{i}", data=frame, compression="gzip")
            hf.create_dataset(f"world_coord_{i}", data=world_coord, compression="gzip")
        # print("Frame shape:", all_frames.shape)
        # print("World Coord shape:", all_world_coord.shape)
        # hf.create_dataset("frames", data=all_frames, compression="gzip")
        # hf.create_dataset("world_coord", data=all_world_coord, compression="gzip")
        # hf.create_dataset("robot", data=robot, compression="gzip")
        # hf.create_dataset("actions", data=actions, compression="gzip")

    # print out stats about the dataset
    stats_str = f"Avg len: {np.mean(len_stats)}\nstd: {np.std(len_stats)}\nmin: {np.min(len_stats)}\nmax: {np.max(len_stats)}"
    print(stats_str)
    stats_path = os.path.join(config.demo_dir, f"stats_{behavior}_{config.seed}.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)


def create_demo_dataset(config, num_demo, num_workers, record, behavior, ep_len):
    """
    Collect all demonstrations and save into demo_dir
    You can use multiple workers if generating 1000s of demonstrations
    """
    from multiprocessing import Process

    os.makedirs(config.demo_dir, exist_ok=True)
    if num_workers == 1:
        generate_demos(0, config, behavior, record, num_demo, ep_len)
    else:
        ps = []
        for i in range(num_workers):
            p = Process(
                target=generate_demos,
                args=(i, config, behavior, record, num_demo, ep_len),
            )
            ps.append(p)

        for p in ps:
            p.start()

        for p in ps:
            p.join()


def collect_push_data():
    num_workers = 1
    num_push = 10
    record = False
    ep_len = 12  # gonna be off by 1 because of reset but whatever

    config, _ = argparser()
    config.norobot_pixels_ob = False
    config.depth_ob = True
    config.reward_type = "dense"
    config.demo_dir = "demos/tckn_data"
    config.most_recent_background = False
    config.multiview = True
    config.img_dim = 128
    config.camera_ids = [0, 1]
    config.temporal_beta = 0.3  # control random policy's temporal correlation
    config.action_noise = 0.5
    # create_demo_dataset(config, num_push, num_workers, record, "straight_push", ep_len)
    create_demo_dataset(
        config, num_push, num_workers, record, "temporal_random_robot", ep_len
    )


if __name__ == "__main__":
    """
    Collect pushing dataset for TCKN probject.
    """
    collect_push_data()
