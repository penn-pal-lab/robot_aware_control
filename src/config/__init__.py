import argparse
from argparse import ArgumentParser

def str2bool(v):
    return v.lower() == "true"


def str2intlist(value):
    if not value:
        return value
    else:
        return [int(num) for num in value.split(",")]


def str2list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(",")]


def create_parser():
    """
    Creates the argparser.  Use this to add additional arguments
    to the parser later.
    """
    parser = argparse.ArgumentParser(
        "Robot Aware Cost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--jobname", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_entity", type=str, default="pal")
    parser.add_argument("--wandb_project", type=str, default="roboaware")
    add_method_arguments(parser)

    return parser


def add_method_arguments(parser: ArgumentParser):
    # method arguments
    parser.add_argument(
        "--reward_type",
        type=str,
        default="weighted",
        choices=["weighted", "dense", "inpaint", "sparse" "blackrobot", "inpaint-blur"],
    )
    parser.add_argument("--robot_pixel_weight", type=float, default=0)
    # inpaint-blur
    parser.add_argument("--blur_sigma", type=float, default=10)
    parser.add_argument("--unblur_cost_scale", type=float, default=3)
    # switch at step L - unblur_timestep
    parser.add_argument("--unblur_timestep", type=float, default=1)

    # control algorithm
    parser.add_argument(
        "--mbrl_algo",
        type=str,
        default="cem",
        choices=["cem"],
    )

    # training
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--record_trajectory", type=str2bool, default=False)
    parser.add_argument("--record_trajectory_interval", type=int, default=5)
    parser.add_argument("--record_video_interval", type=int, default=1)

    # environment
    parser.add_argument("--env", type=str, default="FetchPush")
    args, unparsed = parser.parse_known_args()

    add_prediction_arguments(parser)
    add_dataset_arguments(parser)

    if args.mbrl_algo == "cem":
        add_cem_arguments(parser)

    # env specific args
    if args.env == "FetchPush":
        add_fetch_push_arguments(parser)

    return parser


# Env Hyperparameters
def add_fetch_push_arguments(parser: ArgumentParser):
    # override prediction dimension stuff
    parser.set_defaults(robot_dim=6, robot_enc_dim=6)
    parser.add_argument("--img_dim", type=int, default=128)
    parser.add_argument(
        "--camera_name",
        type=str,
        default="external_camera_0",
        choices=["head_camera_rgb", "gripper_camera_rgb", "lidar", "external_camera_0"],
    )
    parser.add_argument("--multiview", type=str2bool, default=False)
    parser.add_argument("--camera_ids", type=str2intlist, default=[0, 4])
    parser.add_argument("--pixels_ob", type=str2bool, default=True)
    parser.add_argument("--norobot_pixels_ob", type=str2bool, default=False)
    parser.add_argument("--object_dist_threshold", type=float, default=0.01)
    parser.add_argument("--gripper_dist_threshold", type=float, default=0.025)
    parser.add_argument("--push_dist", type=float, default=0.2)
    parser.add_argument("--max_episode_length", type=int, default=10)
    parser.add_argument(
        "--robot_goal_distribution",
        type=str,
        default="random",
        choices=["random", "behind_block"],
    )
    parser.add_argument("--large_block", type=str2bool, default=False)
    parser.add_argument("--invisible_demo", type=str2bool, default=False)
    parser.add_argument("--demo_dir", type=str, default="demos/fetch_push")


# Video Prediction arguments from SVG
def add_prediction_arguments(parser):
    parser.add_argument("--lr", default=0.002, type=float, help="learning rate")
    parser.add_argument(
        "--beta1", default=0.9, type=float, help="momentum term for adam"
    )
    parser.add_argument("--batch_size", default=100, type=int, help="batch size")
    parser.add_argument("--optimizer", default="adam", help="optimizer to train with")
    parser.add_argument(
        "--niter", type=int, default=300, help="number of epochs to train for"
    )
    parser.add_argument("--epoch_size", type=int, default=600, help="epoch size")
    parser.add_argument(
        "--image_width",
        type=int,
        default=128,
        help="the height / width of the input image to network",
    )
    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--dataset", default="smmnist", help="dataset to train with")
    parser.add_argument(
        "--n_past", type=int, default=1, help="number of frames to condition on"
    )
    parser.add_argument(
        "--n_future",
        type=int,
        default=9,
        help="number of frames to predict during training",
    )
    parser.add_argument(
        "--n_eval", type=int, default=10, help="number of frames to predict during eval"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    parser.add_argument(
        "--rnn_size", type=int, default=256, help="dimensionality of hidden layer"
    )
    parser.add_argument(
        "--prior_rnn_layers", type=int, default=1, help="number of layers"
    )
    parser.add_argument(
        "--posterior_rnn_layers", type=int, default=1, help="number of layers"
    )
    parser.add_argument(
        "--predictor_rnn_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--z_dim", type=int, default=10, help="dimensionality of z_t")
    parser.add_argument(
        "--g_dim",
        type=int,
        default=128,
        help="dimensionality of encoder output vector and decoder input vector",
    )
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--action_enc_dim", type=int, default=3)

    parser.add_argument("--robot_dim", type=int, default=6)
    parser.add_argument("--robot_enc_dim", type=int, default=6)

    parser.add_argument(
        "--beta", type=float, default=0.0001, help="weighting on KL to prior"
    )
    parser.add_argument("--model", default="vgg", help="model type (dcgan | vgg)")

    parser.add_argument(
        "--last_frame_skip",
        type=str2bool,
        default=False,
        help="if true, skip connections go between frame t and frame t+t rather than last ground truth frame",
    )


def add_dataset_arguments(parser):
    parser.add_argument(
        "--data_threads", type=int, default=5, help="number of data loading threads"
    )
    parser.add_argument("--data_root", default="data", help="root directory for data")
    parser.add_argument("--train_val_split", type=float, default=0.8)


# Algo Hyperparameters
def add_cem_arguments(parser):
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--opt_iter", type=int, default=10)
    parser.add_argument("--action_candidates", type=int, default=30)
    parser.add_argument("--topk", type=int, default=5)


def argparser():
    """ Directly parses the arguments. """
    parser = create_parser()
    args, unparsed = parser.parse_known_args()
    assert len(unparsed) == 0
    return args, unparsed
