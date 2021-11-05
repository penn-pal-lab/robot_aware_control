import logging
import os
import pickle
from collections import defaultdict
from os.path import join

from src.cem.pick.cem import CEMPolicy
from src.utils.mujoco import init_mjrender_device
import colorlog
import h5py
import imageio
import numpy as np
import torch
import wandb
from tqdm import trange
from src.utils.state import State, DemoGoalState
# from src.env.robotics.locobot_pick_env import LocobotPickEnv
from src.env.robotics.locobot_pick_env_mv import LocobotPickEnv
from src.prediction.losses import RobotWorldCost


class EpisodeRunner(object):
    """
    Run the demonstration following episodes and logs the metrics
    """

    def __init__(self, config) -> None:
        self._config = config
        self._use_env = config.use_env_dynamics
        self._setup_loggers(config)
        init_mjrender_device(config)
        self._env = LocobotPickEnv(config)
        self._setup_policy(config, self._env)
        self._setup_cost(config)
        self._stats = defaultdict(list)

    def _setup_policy(self, config, env):
        self.policy: CEMPolicy = CEMPolicy(
            config,
            physics="gt" if config.use_env_dynamics else "learned",
            init_std=config.cem_init_std,
            action_candidates=config.action_candidates,
            horizon=config.horizon,
            opt_iter=config.opt_iter
        )

    def _setup_cost(self, config):
        self.cost: RobotWorldCost = RobotWorldCost(config)

    def run_episode(self, ep_num, demo_name, demo_path):
        """
        Run one demo-following episode
        """
        cfg = self._config
        env = self._env
        logger = self._logger
        logger.info(f"=== Episode {ep_num}, {demo_name} ===")
        ep_stats = defaultdict(list)
        trajectory = defaultdict(list)

        demo = self._load_demo(demo_path)
        if cfg.cyclegan: # load goals from source robot
            source_demo_path = "/home/edward/roboaware/demos/locobot_pick_demos/none_pick_0_2_s.hdf5"
            source_demo = self._load_demo(source_demo_path)
            goal_timesteps = DEMO_GOAL_LABELS["none_pick_0_2_s.hdf5"]
            goals = self._extract_goals_from_demo(goal_timesteps, source_demo)
        else:
            goal_timesteps = DEMO_GOAL_LABELS[demo_name]
            goals = self._extract_goals_from_demo(goal_timesteps, demo)

        max_timestep = 16
        start_timestep = 2
        initial_state = {"qpos": demo["qpos"][start_timestep], "obj_qpos": demo["obj_qpos"][start_timestep], "states": demo["eef_states"][start_timestep]}
        goal_pos = demo["obj_qpos"][-1][:3]

        if cfg.cyclegan:
            initial_state["obj_qpos"] = source_demo["obj_qpos"][start_timestep]
        obs = env.reset(initial_state=initial_state, init_robot_qpos=True)
        if cfg.cyclegan:
            obs["real_observation"] = obs["observation"]
            obs["observation"] = self._cyclegan_forward(obs["observation"])
        trajectory["obs"].append(obs)
        initial_obj_z = obs["obj_qpos"][2]
        init_obj_pos = obs["obj_qpos"][:3]
        obj_dist = init_obj_dist = np.linalg.norm(init_obj_pos - goal_pos)
        gripper_dist = np.linalg.norm(obs["obj_qpos"][:3] - env.get_gripper_world_pos())
        print("initial obj-goal dist", obj_dist)
        print("initial obj-gripper dist", gripper_dist)

        goal_idx = 0

        # begin policy rollout
        success = False
        obj_picked = False
        ep_timestep = start_timestep
        terminate_ep = False
        steps_per_goal = cfg.horizon
        # before_img = env._get_obs()["observation"]
        # imageio.imwrite("before_ac.png", before_img)
        while True:
            curr_img = obs["observation"]
            curr_mask = obs["masks"]
            curr_state = obs["states"]
            curr_sim_state = None
            curr_sim_state = env.get_flattened_state().copy()
            start = State(img=curr_img, state=curr_state, mask=curr_mask, sim_state=curr_sim_state)

            goal_timestep = goal_timesteps[goal_idx]
            curr_goal = goals[goal_idx]
            goal_masks = None
            if cfg.goal_image_type == "image":
                goal_imgs = [curr_goal["observations"]]
                goal_masks = [curr_goal["masks"]]
            elif cfg.goal_image_type == "object_only":
                goal_imgs = [curr_goal["obj_observations"]]
                goal_masks = [np.zeros_like(curr_goal["masks"])]

            goal_eef_states = [curr_goal["eef_states"]]
            opt_traj = demo["actions"][max(0, goal_timestep - cfg.horizon) : goal_timestep]
            goal = DemoGoalState(imgs=goal_imgs, masks=goal_masks, states=goal_eef_states)

            # figure out horizon
            self.policy.horizon = min(cfg.horizon, steps_per_goal)
            print("setting horizon to", self.policy.horizon)
            ac = self.policy.get_action(start, goal, ep_num, ep_timestep, opt_traj)[0]
            print("executing", ac)
            trajectory["ac"].append(ac)
            obs, _, _, _ = env.step(ac)
            if cfg.cyclegan:
                obs["real_observation"] = obs["observation"]
                obs["observation"] = self._cyclegan_forward(obs["observation"])
            trajectory["obs"].append(obs)

            ep_timestep += 1
            steps_per_goal -= 1
            obj_dist = np.linalg.norm(obs["obj_qpos"][:3] - goal_pos)
            print("step", ep_timestep, obj_dist)
            success = obj_dist < 0.01
            if np.linalg.norm(init_obj_pos - obs["obj_qpos"][:3]) < 0.02 and ep_timestep > 8:
                print("obj didn't move enough, early exiting")
                break
            if (obj_dist - 0.03) >= init_obj_dist:
                print("obj moved away from goal, early exiting")
                break
            if ep_timestep >= max_timestep or obj_dist < 0.015:
                break

            # TODO: update goal timestep based on progress
            curr_state = State(img=obs["observation"], mask=obs["masks"], state=obs["states"])
            goal_state =  State(img=goal_imgs[0], mask=goal_masks[0], state=goal_eef_states[0])

            if self._advance_to_next_goal(curr_state, goal_state, goal_idx):
                if goal_idx < len(goals):
                    goal_idx += 1
                print("advancing to next goal", goal_idx)
                steps_per_goal = cfg.horizon

            if goal_idx == len(goals):
                print("done with all subgoals")
                break

            if steps_per_goal < 2:
                print("ran out of steps for achieving goal", goal_idx)
                break

        if success:
            logger.info("SUCCESS")

        record_path = join(cfg.log_dir, f"exec_{ep_num}.gif")
        if cfg.cyclegan:
            imageio.mimwrite(record_path, [np.concatenate([x["observation"], x["real_observation"]], 1) for x in trajectory["obs"]], fps=4)
        else:
            imageio.mimwrite(record_path, [x["observation"] for x in trajectory["obs"]], fps=4)
        record_path = join(cfg.log_dir, f"true_exec_{ep_num}.gif")
        imageio.mimwrite(record_path, demo["observations"][start_timestep:], fps=4)
        ep_stats["success"] = success
        ep_stats["final_dist"] = obj_dist
        return ep_stats

    def _advance_to_next_goal(self, curr: State, goal: State, goal_idx=None):
        cfg = self._config
        if not hasattr(self, "_old_robot_cost_success"):
            self._old_robot_cost_success = cfg.robot_cost_success
        robot_success = True
        print_str = "Checking goal, "
        if cfg.robot_cost_weight != 0:
            eef_dist = -1 * self.cost.robot_cost(curr, goal)
            if goal_idx == 1: # relax lift subgoal
                cfg.robot_cost_success = 0.025
            else:
                cfg.robot_cost_success = self._old_robot_cost_success
            robot_success = eef_dist < cfg.robot_cost_success
            print_str += f"eef dist: {eef_dist}, {robot_success}"

        world_success = True
        if cfg.world_cost_weight != 0:
            img_dist = -1 * self.cost.world_cost(curr, goal)
            world_success = img_dist < cfg.world_cost_success
            print_str += f" , world dist: {img_dist}, {world_success}"

        all_success = robot_success and world_success
        print(print_str)
        return all_success

    def _extract_goals_from_demo(self, timesteps, demo):
        """Get a specific subset of goals"""
        goals = []
        for t in timesteps:
            goal = {}
            for k, v in demo.items():
                if k != "actions":
                    goal[k] = v[t]
            goals.append(goal)
        return goals

    def run(self):
        """
        Run all episodes and log their metrics
        """
        trials_per_demo = 5
        files = self._load_demo_dataset(self._config)
        all_files = []
        success = 0
        final_dist = 0
        all_stats = []
        for f in files:
            all_files.extend([f for _ in range(trials_per_demo)])

        for i in trange(len(all_files), desc="running episode"):
            demo_name, demo_path = all_files[i]
            ep_stats = self.run_episode(i, demo_name, demo_path)
            all_stats.append(ep_stats)
            success += ep_stats["success"]
            final_dist += ep_stats["final_dist"]
            print(f"running success {success}/{i+1}",  success / (i + 1), f" dist", final_dist / (i+1))

        self._logger.info("\n\n### Summary ###")
        self._logger.info(f"Success {success}/{len(all_files)},   {(success / len(all_files)):.2f}%")
        self._logger.info(f"Final Dist {(final_dist / len(all_files)):.2f}")

        with open(join(self._config.log_dir, "stats.pkl"), "wb") as f:
            pickle.dump(all_stats, f)

    def _load_demo_dataset(self, config):
        files = []
        for d in os.scandir(config.object_demo_dir):
            if d.is_file() and d.name.endswith("hdf5"):
                files.append((d.name, d.path))
        files.sort(key=lambda e: e[0])
        return files[: config.num_episodes]

    def _load_demo(self, path):
        # load start,
        demo = {}
        with h5py.File(path, "r") as hf:
            demo["obj_qpos"] = hf["obj_qpos"][:]
            demo["qpos"] = hf["qpos"][:]
            demo["eef_states"] = hf["states"][:]
            demo["observations"] = hf["observations"][:]
            demo["obj_observations"] = hf["obj_only_imgs"][:]
            demo["masks"] = hf["masks"][:]
            demo["actions"] = hf["actions"][:][:, (0,1,2,4)]
        return demo

    def load_cyclegan(self, opt):
        from src.cyclegan.models import create_model
        self.cyclegan = create_model(opt)
        self.cyclegan.setup(opt)
        self.cyclegan.eval()

    def _cyclegan_forward(self, input_img):
        """
        Assumes img is a np.uint8 numpy array of dim [64, 48, 3] .
        """
        import torchvision.transforms as transforms
        oh, ow, = input_img.shape[:2]
        transform_A = get_transform(opt, grayscale=False)
        input_img = Image.fromarray(np.uint8(input_img)).convert("RGB")
        img = transform_A(input_img).unsqueeze(0) # becomes [1,256, 256, 3]
        with torch.no_grad():
            output = self.cyclegan.netG(img).cpu()
        # output is (1,3, 64, 48) and normalized between -1 and 1. scale back to [0,1]
        output_img = (output[0] + 1) / 2
        # then resize img back to (64, 48)
        resize = transforms.Resize((oh, ow), Image.BICUBIC)
        output_img = resize(output_img)
        # then make image between [0, 255] and uint8
        output_img = np.uint8(255 * output_img.numpy())
        output_img = output_img.transpose((1,2,0)) # put Channel dim at back
        return output_img

    def _setup_loggers(self, config):
        # make folder for exp logs
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red,bold",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
        # only logs to console
        logger = colorlog.getLogger("console")
        logger.setLevel(logging.DEBUG)

        ch = colorlog.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        config.log_dir = join(config.log_dir, config.jobname)
        logger.info(f"Create log directory: {config.log_dir}")
        os.makedirs(config.log_dir, exist_ok=True)

        config.plot_dir = join(config.log_dir, "plot")
        os.makedirs(config.plot_dir, exist_ok=True)

        config.video_dir = join(config.log_dir, "video")
        os.makedirs(config.video_dir, exist_ok=True)

        config.trajectory_dir = join(config.log_dir, "trajectory")
        os.makedirs(config.trajectory_dir, exist_ok=True)

        # create the file / console logger
        filelogger = colorlog.getLogger("file/console")
        filelogger.setLevel(logging.DEBUG)
        logfile_path = join(config.log_dir, "log.txt")
        fh = logging.FileHandler(logfile_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s @l%(lineno)d: %(message)s", "%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        filelogger.addHandler(fh)
        filelogger.addHandler(ch)
        self._logger = colorlog.getLogger("file/console")

        # device
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        config.device = device

        # wandb stuff
        if not config.wandb:
            os.environ["WANDB_MODE"] = "dryrun"
        os.environ["WANDB_API_KEY"] = "24e6ba2cb3e7bced52962413c58277801d14bba0"
        exclude = ["device"]
        wandb.init(
            resume=config.jobname,
            project=config.wandb_project,
            config={k: v for k, v in config.__dict__.items() if k not in exclude},
            dir=config.log_dir,
            entity=config.wandb_entity,
        )

def visualize_demo(demo_path):
    demo = {}
    with h5py.File(demo_path, "r") as hf:
        demo["obj_qpos"] = hf["obj_qpos"][:]
        demo["qpos"] = hf["qpos"][:]
        demo["eef_states"] = hf["states"][:]
        demo["observations"] = hf["observations"][:]
        demo["obj_observations"] = hf["obj_only_imgs"][:]
        demo["masks"] = hf["masks"][:]
        demo["actions"] = hf["actions"][:][:, (0,1,2,4)]

    for i, obs in enumerate(demo["observations"]):
        imageio.imwrite(f"obs_{i}.png", obs)

if __name__ == "__main__":
    import sys
    from src.config import argparser

    # demo_path = "/home/edward/roboaware/demos/fetch_pick_demos/none_pick_0_6_f.hdf5"
    demo_path = "/home/edward/roboaware/demos/labeled_fetch_pick_mv_demos/none_pick_0_9_f.hdf5"
    # demo_path = "/home/edward/roboaware/demos/locobot_pick_demos/none_pick_0_2_s.hdf5"
    demo_name = "none_pick_0_9_f"
    config, _ = argparser()
    # visualize_demo(demo_path)
    # sys.exit(0)

    # env
    config.modified = True
    config.object_demo_dir = "/home/edward/roboaware/demos/labeled_fetch_pick_mv_demos"
    # config.object_demo_dir = "/home/pallab/roboaware/demos/fetch_pick_demos"
    # config.object_demo_dir = "/home/edward/roboaware/demos/locobot_pick_demos"

    # Fetch Pick demos
    DEMO_GOAL_LABELS = {
        "none_pick_0_5_s.hdf5": [3,9,12],
        "none_pick_0_6_f.hdf5": [3,8,12],
        "none_pick_0_8_s.hdf5": [6,11,14],
        "none_pick_0_9_f.hdf5": [4,9,12],
    }

    # Locobot Pick demos
    # DEMO_GOAL_LABELS = {
    #     "none_pick_0_2_s.hdf5": [4,9,13],
    # }

    # model
    # config.dynamics_model_ckpt = "/home/edward/roboaware/checkpoints/pick4_ranofuture_72300.pt"
    config.action_dim = 5
    # config.action_dim = 5
    # config.robot_dim = 5
    # config.model_use_robot_state = True
    # config.model_use_mask = True
    # config.model_use_future_mask = False
    # config.model_use_future_robot_state = False
    # config.lstm_group_norm = True
    # config.g_dim = 256
    # config.z_dim  = 64


    # cem
    config.debug_cem = True
    config.action_candidates = 500
    config.candidates_batch_size = 500
    config.use_env_dynamics = False
    config.cem_init_std = 0.5
    config.horizon = 5
    config.opt_iter = 5
    config.topk = 5

    # RA cost
    config.goal_image_type = "object_only" # object only or robot image
    config.reward_type = "dontcare"
    config.sparse_cost = True
    config.robot_cost_weight = 1
    config.world_cost_weight = 0.5
    config.robot_cost_success = 0.015
    config.world_cost_success = 0.5

    # Pixel cost
    # config.goal_image_type = "image" # object only or robot image
    # config.reward_type = "dense"
    # config.sparse_cost = False
    # config.robot_cost_weight = 0
    # config.world_cost_weight = 0.0001
    # config.robot_cost_success = 0.015
    # config.world_cost_success = 1000

    runner = EpisodeRunner(config)
    runner.run()

    # load cyclegan stuff
    # from src.cyclegan.data.base_dataset import get_transform
    # from PIL import Image

    # with open("src/cyclegan/test_config.pkl", "rb") as f:
    #     opt = pickle.load(f)
    # opt.checkpoints_dir = "/home/edward/projects/pytorch-CycleGAN-and-pix2pix/checkpoints"
    # opt.gpu_ids = [1]
    # opt.model = "test" # load only one model
    # opt.model_suffix = "_A"
    # opt.name = "ra_cyclegan_v2"
    # # visualize_demo(demo_path)
    # runner = EpisodeRunner(config)
    # runner.load_cyclegan(opt)
    # # img = np.zeros((64, 48, 3))
    # # runner._cyclegan_forward(img)
    # runner.run()
    # # runner.run_episode(0, demo_name, demo_path)
