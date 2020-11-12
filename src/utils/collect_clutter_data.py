from functools import partial
import os
from collections import defaultdict
from copy import deepcopy

import cv2
import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.config import argparser
from src.env.fetch.clutter_push import ClutterPushEnv
from tqdm import tqdm, trange


def collect_trajectory(rank, config, behavior, record, num_trajectories, ep_len):
    config.seed = rank
    env = ClutterPushEnv(config)
    len_stats = []
    it = range(num_trajectories)
    if rank == 0:
        it = tqdm(it)
    for i in it:
        # only record first episode for sanity check
        record = rank == 0 and record
        name = f"{behavior}_{rank}_{i}.hdf5"
        path = os.path.join(config.demo_dir, name)
        history = env.generate_demo(behavior, ep_len=ep_len)
        pushed_obj = history["pushed_obj"]
        record_path = f"videos/{behavior}_{config.seed}_{i}.gif"
        obs = history["obs"]  # array of observation dictionaries
        len_stats.append(len(obs))
        object_inpaint_demo = []
        robot = []
        obj_poses = defaultdict(list)
        states = []
        for ob in obs:
            object_inpaint_demo.append(ob["observation"])
            states.append(ob["state"])
            for obj in env._objects:
                obj_poses[obj + ":joint"].append(ob[obj + ":joint"])
        object_inpaint_demo = np.asarray(object_inpaint_demo)
        robot = np.asarray(robot)
        actions = history["ac"]
        assert len(object_inpaint_demo) - 1 == len(actions)
        # now render the object only demonstration
        env = ClutterPushEnv(config)
        env.reset()
        # first move robot out of view
        env.sim.data.set_joint_qpos("robot0:slide2", 1)
        env.sim.data.set_joint_qpos("robot0:slide0", -1)
        env.sim.forward()
        # save the arm overhead state for rendering object only scene
        arm_overhead_state = env.get_flattened_state()
        # for each timestep, set objects, render the image, and save
        object_only_demo = []
        robot_demo = []
        gif = []
        for t in range(len(object_inpaint_demo)):
            for k, v in obj_poses.items():
                env.sim.data.set_joint_qpos(k, v[t])
            env.sim.forward()
            object_only_img = env.render()
            object_only_demo.append(object_only_img)
            # now render the with robot img
            env.set_flattened_state(states[t])
            robot_img = env.render(remove_robot=False)
            robot_demo.append(robot_img)
            if record:
                # concat with the demo's inpainted image and background img
                putText = partial(
                    cv2.putText,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                gif_robot_img = robot_img.copy()
                putText(gif_robot_img, f"REAL", (10, 10))
                putText(gif_robot_img, f"{t}", (0, 126))

                gif_inpaint_img = object_inpaint_demo[t].copy()
                putText(gif_inpaint_img, f"INPAINT", (10, 10))

                gif_object_only_img = object_only_img.copy()
                putText(gif_object_only_img, f"NO-ROBOT", (10, 10))

                img = np.concatenate(
                    [gif_robot_img, gif_inpaint_img, gif_object_only_img], axis=1
                )
                gif.append(img)
            env.set_flattened_state(arm_overhead_state)

        if record:
            imageio.mimwrite(record_path, gif)
        with h5py.File(path, "w") as hf:
            hf.attrs["pushed_obj"] = str(history["pushed_obj"])
            hf.create_dataset("states", data=states, compression="gzip")
            hf.create_dataset("actions", data=actions, compression="gzip")
            # ground truth object demo
            hf.create_dataset(
                "object_only_demo", data=object_only_demo, compression="gzip"
            )
            # inpainted object demo
            hf.create_dataset(
                "object_inpaint_demo", data=object_inpaint_demo, compression="gzip"
            )
            # with robot demo
            hf.create_dataset("robot_demo", data=robot_demo, compression="gzip")
            for obj in env._objects:
                hf.create_dataset(
                    obj + ":joint", data=obj_poses[obj + ":joint"], compression="gzip"
                )

    # print out stats about the dataset
    stats_str = f"Avg len: {np.mean(len_stats)}\nstd: {np.std(len_stats)}\nmin: {np.min(len_stats)}\nmax: {np.max(len_stats)}"
    print(stats_str)
    stats_path = os.path.join(config.demo_dir, f"stats_{behavior}_{config.seed}.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)


def collect_trajectories():
    """
    Collect various robot skill demonstrations
    """
    from multiprocessing import Process

    num_trajectories = 100  # per worker
    num_workers = 1
    record = True
    behavior = "straight_push"
    ep_len = 12  # gonna be off by -1 because of reset but whatever

    config, _ = argparser()
    config.norobot_pixels_ob = True
    config.reward_type = "inpaint"
    config.demo_dir = "demos/straight_push"
    config.most_recent_background = False
    config.multiview = True
    config.img_dim = 64
    config.camera_ids = [0, 1]

    os.makedirs(config.demo_dir, exist_ok=True)
    if num_workers == 1:
        collect_trajectory(0, config, behavior, record, num_trajectories, ep_len)
    else:
        ps = []
        for i in range(num_workers):
            p = Process(
                target=collect_trajectory,
                args=(i, config, behavior, record, num_trajectories, ep_len),
            )
            ps.append(p)

        for p in ps:
            p.start()

        for p in ps:
            p.join()


def collect_multiview_trajectory(
    rank, config, behavior, record, num_trajectories, ep_len
):
    # save the background image for inpainting?
    # save the robot segmentation mask?
    # or just save the inpainted image directly?
    config.seed = rank
    env = ClutterPushEnv(config)
    len_stats = []
    it = range(num_trajectories)
    if rank == 0:
        it = tqdm(it)
    for i in it:
        # only record first episode for sanity check
        record = rank == 0 and i == 0 and record
        name = f"{behavior}_{rank}_{i}.hdf5"
        path = os.path.join(config.demo_dir, name)
        record_path = f"videos/{behavior}_{config.seed}_{i}.mp4"
        history = env.generate_demo(
            behavior, record=record, record_path=record_path, ep_len=ep_len
        )
        obs = history["obs"]  # array of observation dictionaries
        len_stats.append(len(obs))
        frames = []
        robot = []
        for ob in obs:
            frames.append(ob["observation"])
            robot.append(ob["robot"])

        frames = np.asarray(frames)
        robot = np.asarray(robot)
        actions = history["ac"]
        assert len(frames) - 1 == len(actions)
        with h5py.File(path, "w") as hf:
            hf.create_dataset("frames", data=frames, compression="gzip")
            hf.create_dataset("robot", data=robot, compression="gzip")
            hf.create_dataset("actions", data=actions, compression="gzip")

    # print out stats about the dataset
    stats_str = f"Avg len: {np.mean(len_stats)}\nstd: {np.std(len_stats)}\nmin: {np.min(len_stats)}\nmax: {np.max(len_stats)}"
    print(stats_str)
    stats_path = os.path.join(config.demo_dir, f"stats_{behavior}_{config.seed}.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)


def collect_multiview_trajectories():
    """
    Collect multiview dataset with inpainting
    """
    from multiprocessing import Process

    num_trajectories = 5000  # per worker
    num_workers = 20
    record = False
    behavior = "random_robot"
    ep_len = 12  # gonna be off by -1 because of reset but whatever

    config, _ = argparser()
    config.large_block = True
    config.demo_dir = "demos/fetch_push_mv"
    config.multiview = True
    config.norobot_pixels_ob = True
    config.img_dim = 64
    os.makedirs(config.demo_dir, exist_ok=True)

    if num_workers == 1:
        collect_multiview_trajectory(
            0, config, behavior, record, num_trajectories, ep_len
        )
    else:
        ps = []
        for i in range(num_workers):
            if i % 2 == 0:
                behavior = "random_robot"
            else:
                behavior = "push"
            p = Process(
                target=collect_multiview_trajectory,
                args=(i, config, behavior, record, num_trajectories, ep_len),
            )
            ps.append(p)

        for p in ps:
            p.start()

        for p in ps:
            p.join()


if __name__ == "__main__":
    # plot_behaviors_per_cost()
    # plot_costs_per_behavior()
    collect_trajectories()
    # collect_multiview_trajectories()
    # collect_cem_goals()
    # collect_object_demos()
    # create_object_demos("demos/straight_push/straight_push_0_9.hdf5")
    # create_heatmap()
