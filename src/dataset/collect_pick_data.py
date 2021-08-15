from functools import partial
import os
from src.utils.mujoco import init_mjrender_device

import h5py
import imageio
import numpy as np
from src.config import argparser
from src.env.robotics.locobot_pick_env import LocobotPickEnv
from tqdm import tqdm


def generate_demos(rank, config, record, num_trajectories, use_noise):
    """ rank: idx of the worker
    config: configuration
    record: whether to record the gif or not
    num_trajectories: number of demos to generate
    ep_len: max length of the demo. only used by some behaviors.
    noise: action noise in behavior. only used by some behaviors
    """
    config.seed = rank
    env = LocobotPickEnv(config)
    len_stats = []
    succ_stats = 0
    it = range(num_trajectories)
    if rank == 0:
        it = tqdm(it)
    for i in it:
        record = rank == 0 and record
        history = env.generate_demo(use_noise)
        name = f"{'noise_' if use_noise else ''}pick_{rank}_{i}_{'s' if history['success'] else 'f'}.hdf5"
        path = os.path.join(config.demo_dir, name)
        obs = history["obs"]  # array of observation dictionaries
        len_stats.append(len(obs))
        succ_stats += int(history["success"])
        states = []
        masks = []
        imgs = []
        qpos = []
        obj_qpos = []
        for ob in obs:
            imgs.append(ob["observation"])
            masks.append(ob["masks"])
            states.append(ob["states"])
            qpos.append(ob["qpos"])
            obj_qpos.append(ob["obj_qpos"])

        # make 3d action 5d for video prediction
        actions = [np.array([ac[0], ac[1], ac[2], 0, ac[3]]) for ac in history["ac"]]
        # now render the object only demonstration
        env.reset()
        # first move robot out of view
        env.sim.data.set_joint_qpos("robot0:slide2", 1)
        env.sim.data.set_joint_qpos("robot0:slide0", -1)
        env.sim.forward()
        # save the arm overhead state for rendering object only scene
        arm_overhead_state = env.get_flattened_state()
        # for each timestep, set objects, render the image, and save
        obj_only_obs = []
        for t in range(len(obs)):
            env.sim.data.set_joint_qpos("object1:joint",obj_qpos[t])
            env.sim.forward()
            object_only_img = env.render()
            obj_only_obs.append(object_only_img)
            env.set_flattened_state(arm_overhead_state)

        if record:
            record_path = f"videos/{'noise_' if use_noise else ''}pick_{config.seed}_{i}.gif"
            imageio.mimwrite(record_path, imgs)
            # record_path = f"videos/pick_{config.seed}_{i}_obj.gif"
            # imageio.mimwrite(record_path, obj_only_obs)

        with h5py.File(path, "w") as hf:
            create_dataset = partial(hf.create_dataset, compression="gzip")
            create_dataset("actions", data=actions)
            create_dataset("observations", data=imgs)
            create_dataset("states", data=states)
            create_dataset("qpos", data=qpos)
            create_dataset("obj_qpos", data=obj_qpos)
            create_dataset("masks", data=masks)
            create_dataset("obj_only_imgs", data=obj_only_obs)

    # print out stats about the dataset
    stats_str = f"Avg len: {np.mean(len_stats)}\nstd: {np.std(len_stats)}\nmin: {np.min(len_stats)}\nmax: {np.max(len_stats)}\n Success: {succ_stats/num_trajectories}"
    print(stats_str)
    stats_path = os.path.join(config.demo_dir, f"{'noise_' if use_noise else ''}stats_pick_{config.seed}.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)


def create_demo_dataset(config, num_demo, num_workers, record, use_noise):
    """
    Collect all demonstrations and save into demo_dir
    You can use multiple workers if generating 1000s of demonstrations
    """
    from multiprocessing import Process

    os.makedirs(config.demo_dir, exist_ok=True)
    if num_workers == 1:
        generate_demos(0, config, record, num_demo, use_noise)
    else:
        ps = []
        for i in range(num_workers):
            p = Process(
                target=generate_demos,
                args=(i, config,record, num_demo, use_noise),
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
    num_workers = 4
    num_demos = 10000 // num_workers
    record = False
    MODIFIED = False

    config, _ = argparser()
    config.gpu = 0
    init_mjrender_device(config)

    config.modified = MODIFIED
    config.demo_dir = f"/scratch/edward/Robonet/locobot_pick{'_fetch' if MODIFIED else ''}_views/c0"
    # config.demo_dir = f"/home/pallab/locobot_ws/src/roboaware/demos/locobot_pick"
    #config.demo_dir = f"/home/ed/roboaware/demos/fetch_pick"
    create_demo_dataset(config, num_demos, num_workers, record, use_noise=True)

    num_demos = 2000 // num_workers
    create_demo_dataset(config, num_demos, num_workers, record, use_noise=False)


if __name__ == "__main__":
    """
    Use this to collect demonstrations for svg / demo cem experiments
    """
    collect_svg_data()
