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

def rollout(history, path):
    frames = history["frame"]
    rewards = history["reward"]
    fig = plt.figure()
    rewards = -1 * np.array([0] + rewards)
    cols = len(frames)
    for n, (image, reward) in enumerate(zip(frames, rewards)):
        a = fig.add_subplot(2, cols, n + 1)
        imagegoal = np.concatenate([image, history["goal"]], axis=1)
        a.imshow(imagegoal)
        a.set_aspect("equal")
        # round reward to 2 decimals
        rew = f"{reward:0.2f}" if n > 0 else "Cost:"
        a.set_title(rew, fontsize=50)
        a.set_xticklabels([])
        a.set_xticks([])
        a.set_yticklabels([])
        a.set_yticks([])
        a.set_xlabel(f"step {n}", fontsize=40)
        # add goal img under every one
        # b = fig.add_subplot(2, cols, n + len(frames) + 1)
        # b.imshow(history["goal"])
        # b.set_aspect("equal")
        # obj =  f"{objd:0.3f}" if n > 0 else "Object Dist:"
        # b.set_title(obj, fontsize=50)
        # b.set_xticklabels([])
        # b.set_xticks([])
        # b.set_yticklabels([])
        # b.set_yticks([])
        # b.set_xlabel(f"goal", fontsize=40)

    fig.set_figheight(10)
    fig.set_figwidth(100)

    # title = f"{title_dict[self.reward_type]} with {behavior} behavior"
    # fig.suptitle(title, fontsize=50, fontweight="bold")
    fig.savefig(path)
    fig.clf()
    plt.close("all")

def collect_cem_goals():
    """Collect goal images for testing CEM planning"""
    config, _ = argparser()
    config.large_block = True
    config.demo_dir = "demos/cem_goals"
    config.multiview = True
    config.norobot_pixels_ob = True
    config.reward_type = "inpaint"
    config.img_dim = 64
    config.push_dist = 0.135  # with noise, (0.07, 0.2)
    os.makedirs(config.demo_dir, exist_ok=True)
    env = ClutterPushEnv(config)
    for i in range(200):
        img = env.reset()
        img_path = os.path.join(config.demo_dir, f"{i}.png")
        imageio.imwrite(img_path, img)


def create_object_demos(config, push_demo_path, gif_path=None, heatmap_path=None):
    """
    Given a robot pushing demo, we now move the robot up, and set the
    object pose for each timestep in the demo
    """

    env = ClutterPushEnv(config)
    env.reset()
    # first move robot out of view
    robot_pos = env.sim.data.get_site_xpos("robot0:grip").copy()
    gripper_target = robot_pos + np.array([-1, 0, 0.5])
    gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
    env.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
    env.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
    for _ in range(10):
        env.sim.step()
    env.sim.forward()
    # save the arm overhead state for rendering object only scene
    arm_overhead_state = env.get_flattened_state()
    # now render the given demonstration
    obj_poses = defaultdict(list)
    ep_len = 0
    demo_frames = []
    with h5py.File(push_demo_path, "r") as hf:
        # demo_frames = hf["frames"][:]
        demo_states = hf["states"][:]
        for obj in env._objects:
            obj_poses[obj + ":joint"] = hf[obj + ":joint"][:]
            ep_len = hf[obj + ":joint"].shape[0]
    # for each timestep, set objects, render the image, and save
    object_demo = []
    demo_imgs = []
    scene_imgs = []
    object_imgs = []
    for t in range(ep_len):
        for k, v in obj_poses.items():
            env.sim.data.set_joint_qpos(k, v[t])
        env.sim.forward()
        object_only_img = env.render()
        object_imgs.append(object_only_img)
        demo_imgs.append(object_only_img)
        # now render the inpaint scene
        env.set_flattened_state(demo_states[t])
        if config.most_recent_background and t == 0:
            # refresh the background img for most recent bg img
            env._background_img = env._get_background_img()
        inpaint_img = env.render(remove_robot=env._norobot_pixels_ob)
        scene_imgs.append(inpaint_img)
        # concat with the demo's inpainted image and background img
        img = np.concatenate(
            [inpaint_img, object_only_img, env._background_img], axis=1
        )
        object_demo.append(img)
        env.set_flattened_state(arm_overhead_state)

    if gif_path is not None:
        imageio.mimwrite(gif_path, object_demo)

    # calculate pairwise distance matrix between inpaint scene and object demo
    if heatmap_path is None:
        return
    skip = 2
    pairwise_cost = np.zeros((ep_len // skip, ep_len // skip), dtype=np.float32)
    for i in range(0, ep_len // skip):
        # row will be demonstration
        for j in range(0, ep_len // skip):
            # columns will be scene
            inpaint_img = scene_imgs[j * skip]
            demo_img = demo_imgs[i * skip]
            # since it's already inpainted, we can just do l2 norm
            pairwise_cost[i, j] = np.linalg.norm(inpaint_img - demo_img)
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        pairwise_cost,
        linewidth=0.5,
        fmt="6.0f",
        annot=True,
        annot_kws={"fontsize": 10},
        square=True,
    )
    plt.ylabel("demo idx")
    plt.xlabel("scene idx")
    plt.title(heatmap_path.split(".")[0])
    plt.savefig(heatmap_path, dpi=200)


def create_heatmap():
    # first create settings
    config, _ = argparser()
    config.multiview = True
    config.img_dim = 64
    config.norobot_pixels_ob = True
    # rewards = ["dontcare", "l2", "inpaint", "mr_inpaint"]
    rewards = ["mr_inpaint"]
    demos = ["demos/straight_push/straight_push_0_8.hdf5"]
    for r in rewards:
        cfg = deepcopy(config)
        if r == "noinpaint":
            cfg.norobot_pixels_ob = False
        elif r == "inpaint":
            cfg.reward_type = "inpaint"
        elif r == "mr_inpaint":
            cfg.reward_type = "inpaint"
            cfg.most_recent_background = True
        for i, d in enumerate(demos):
            gif_path = f"{r}_demo{i}.gif"
            heatmap_path = f"{r}_{i}_heatmap.png"
            create_object_demos(cfg, d, gif_path, heatmap_path)


def collect_inpaint_trajectory(
    rank, config, behavior, record, num_trajectories, ep_len
):
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
        masks = []
        for ob in obs:
            object_inpaint_demo.append(ob["observation"])
            states.append(ob["state"])
            masks.append(ob["mask"])
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
        gif = []
        for t in range(len(object_inpaint_demo)):
            for k, v in obj_poses.items():
                env.sim.data.set_joint_qpos(k, v[t])
            env.sim.forward()
            object_only_img = env.render()
            object_only_demo.append(object_only_img)
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
                gif_mask_img = np.stack((masks[t],) * 3, axis=-1).astype(np.uint8)
                gif_mask_img *= 255

                gif_inpaint_img = object_inpaint_demo[t].copy()
                putText(gif_inpaint_img, f"INPAINT", (10, 10))

                gif_object_only_img = object_only_img.copy()
                putText(gif_object_only_img, f"NO-ROBOT", (10, 10))

                img = np.concatenate(
                    [gif_mask_img, gif_inpaint_img, gif_object_only_img], axis=1
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
            # masks
            hf.create_dataset("masks", data=masks, compression="gzip")

    # print out stats about the dataset
    stats_str = f"Avg len: {np.mean(len_stats)}\nstd: {np.std(len_stats)}\nmin: {np.min(len_stats)}\nmax: {np.max(len_stats)}"
    print(stats_str)
    stats_path = os.path.join(config.demo_dir, f"stats_{behavior}_{config.seed}.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)


def collect_inpaint_trajectories():
    """
    Collect various robot skill demonstrations
    """
    from multiprocessing import Process

    num_trajectories = 1  # per worker
    num_workers = 1
    record = True
    behavior = "straight_push"
    ep_len = 12  # gonna be off by -1 because of reset but whatever

    config, _ = argparser()
    config.norobot_pixels_ob = True
    config.reward_type = "inpaint"
    config.demo_dir = "demos/inpaint_dataset"
    config.most_recent_background = False
    config.multiview = True
    config.img_dim = 64
    config.camera_ids = [0, 1]

    os.makedirs(config.demo_dir, exist_ok=True)
    if num_workers == 1:
        collect_inpaint_trajectory(
            0, config, behavior, record, num_trajectories, ep_len
        )
    else:
        ps = []
        for i in range(num_workers):
            p = Process(
                target=collect_inpaint_trajectory,
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
    # collect_cem_goals()
    # create_heatmap()
    collect_inpaint_trajectories()
