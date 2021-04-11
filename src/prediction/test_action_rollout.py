import numpy as np
import torch
from src.config import argparser
from src.dataset.locobot.locobot_singleview_dataloader import \
    create_transfer_loader
from src.dataset.robonet.robonet_dataset import denormalize, get_batch
from src.prediction.models.dynamics import SVGConvModel
from src.dataset.locobot.locobot_model import LocobotAnalyticalModel
from src.utils.plot import save_gif
from tqdm import trange
import ipdb
from src.utils.camera_calibration import world_to_camera_dict

"""
Apply different actions to the SVG model to see video prediction performance
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zero_robot_region(mask, image, inplace=False):
    """
    Set the robot region to zero
    """
    robot_mask = mask.type(torch.bool)
    robot_mask = robot_mask.repeat(1, 3, 1, 1)
    if not inplace:
        image = image.clone()
    image[robot_mask] *= 0
    return image

def convert_world_to_camera_pos(state, w_to_c):
        e_to_w = np.eye(4)
        e_to_w[:3, 3] = state[:3]
        e_to_c = w_to_c @ e_to_w
        pos_c = e_to_c[:3, 3]
        return pos_c

def load_model(config, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SVGConvModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    return model

@torch.no_grad()
def plot_rollout(model, data, actions, gif_name):
    ''' model rollout '''
    x = data["images"]
    states = data["states"]
    ac = actions
    masks = data["masks"]
    if cf.model_use_heatmap:
        heatmaps = data["heatmaps"]
    robot_name = data["robot"]
    if "finetune" in cf.experiment and (cf.model_use_mask or cf.model_use_robot_state):
        x = data["images"]
        states = data["states"]
        ac = fake_actions
        mask = data["masks"]
        qpos = data["qpos"]
        robot = data["robot"]
        folder = data["folder"]
        input_data = dict(
            states=states,
            actions=ac,
            masks=mask,
            qpos=qpos,
            folder=folder,
            robot=robot,
        )
        input_data["low"] = data["low"]
        input_data["high"] = data["high"]
        if cf.preprocess_action != "raw":
            input_data["raw_actions"] = data["raw_actions"]

        if cf.experiment == "finetune_locobot":
            out = robot_model.predict_batch(input_data)

        if cf.model_use_heatmap:
            states, masks, heatmaps = out
        else:
            states, masks = out

    bs = min(cf.test_batch_size, x.shape[1])
    model.init_hidden(bs)
    robot_name = np.array(robot_name)
    x_pred = skip = None

    gen_seq = [[] for _ in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]
    b = min(x.shape[1], 10)

    for s in trange(nsample):
        skip = None
        model.init_hidden(b)
        if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
            x_j = zero_robot_region(masks[0], x[0])
        else:
            x_j = x[0]
        gen_seq[s].append(x_j)
        for i in range(1, video_len):
            # let j be i - 1, or previous timestep
            m_j, r_j, a_j = masks[i - 1], states[i - 1], ac[i - 1]
            x_i, m_i, r_i = x[i], masks[i], states[i]
            hm_j, hm_i = None, None
            if cf.model_use_heatmap:
                hm_j, hm_i = heatmaps[i - 1], heatmaps[i]
            if cf.model == "copy":
                x_pred = model(x_j, m_j, x_i, m_i)
            else:
                # zero out robot pixels in input for norobot cost
                x_j_black, x_i_black = x_j, x_i
                if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                    x_j_black = zero_robot_region(m_j, x_j)
                    x_i_black = zero_robot_region(m_i, x_i)
                m_in = m_j
                if cf.model_use_future_mask:
                    m_in = torch.cat([m_j, m_i], 1)
                r_in = r_j
                if cf.model_use_future_robot_state:
                    r_in = (r_j, r_i)
                hm_in = hm_j
                if cf.model_use_future_heatmap:
                    hm_in = torch.cat([hm_j, hm_i], 1)

                if cf.last_frame_skip:
                    # overwrite conditioning frame skip if necessary
                    skip = None

                if cf.model == "det":
                    x_pred, curr_skip = model(x_j, m_in, r_j, a_j, skip)
                elif cf.model == "svg":
                    # don't use posterior.
                    x_i_black, m_next_in, r_i, hm_next_in = None, None, None, None
                    out = model(
                        x_j_black,
                        m_in,
                        r_in,
                        hm_in,
                        a_j,
                        x_i_black,
                        m_next_in,
                        r_i,
                        hm_next_in,
                        skip,
                    )
                    x_pred, curr_skip, _, _, _, _ = out

                x_pred, x_pred_mask = x_pred[:, :3], x_pred[:, 3].unsqueeze(1)
                x_pred = (1 - x_pred_mask) * x_j + (x_pred_mask) * x_pred

                if i <= cf.n_past:
                    # store the most recent conditioning frame's skip
                    skip = curr_skip

                if "dontcare" in cf.reconstruction_loss or cf.black_robot_input:
                    zero_robot_region(masks[i], x_pred, inplace=True)
            if i < cf.n_past:
                x_j = x_i
            else:
                x_j = x_pred
            gen_seq[s].append(x_j)

    '''Plot videos'''
    to_plot = []
    gifs = [[] for _ in range(video_len)]
    nrow = b
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(video_len):
            row.append(gt_seq[t][i])
        to_plot.append(row)
        if cf.model == "svg":
            s_list = range(nsample)
        else:
            s_list = [0]
        for t in range(video_len):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)
    # gifs is T x B x S x |I|
    save_gif(gif_name, gifs)

if __name__ == "__main__":
    cf, _ = argparser()
    cf.device = device
    cf.batch_size = 3  # number of videos
    CKPT_PATH = "logs/ft389locobot_norobot_raw_144k_analytical_scheduledsampling/ckpt_10200.pt"
    video_len = 5
    nsample = 3  # number of stochastic samples per video

    model = load_model(cf, CKPT_PATH)
    if cf.experiment == "finetune_locobot":
        robot_model = LocobotAnalyticalModel(cf)
    model.eval()
    ''' load data '''
    loader = create_transfer_loader(cf)
    data_gen = get_batch(loader, device)
    data = next(data_gen)


    ''' decide what actions to take '''
    ALL_DIRECTIONS = [-1, 1]
    ALL_AXES = ["x", "y"]
    STEP_SIZE = 0.02
    world2cam = world_to_camera_dict["locobot_c0"]

    for direction in ALL_DIRECTIONS:
        for ax in ALL_AXES:
            # apply actions (T x B x 5)
            fake_actions = torch.zeros_like(data["actions"])
            fake_actions[:, :, 2] = data["actions"][:, :, 2]  # use same z
            # go {+,-}1.5 cm in the {x,y} direction every step
            if ax == "x":
                axis = 0
            elif ax == "y":
                axis = 1
            else:
                raise ValueError
            fake_actions[:, :, axis] = STEP_SIZE * direction
            # check one action trajectory 
            true_states = data["states"][:, 0].cpu().clone().numpy()
            true_states[:, :3] = denormalize(true_states[:,  :3], data["low"][0,:3].cpu().numpy(), data["high"][0, :3].cpu().numpy())
            old_actions = fake_actions[:, 0].cpu().numpy().copy()
            new_actions = []
            for t in range(len(fake_actions)):
                state = true_states[t]
                pos_c = convert_world_to_camera_pos(state, world2cam)
                next_state = true_states[t].copy()
                next_state[:4] += old_actions[t][:4]
                next_pos_c = convert_world_to_camera_pos(next_state, world2cam)
                true_offset_c = next_pos_c - pos_c
                new_actions.append( true_offset_c)

            
            gif_name = f"{ax}_{int(direction * STEP_SIZE * 1000)}mm.gif"
            print(gif_name)
            print(new_actions[0])
            # plot_rollout(model, data, fake_actions, gif_name)

    ''' diagonal pushing'''
    # ALL_AXES = np.asarray([[1,1], [1, -1], [-1, 1], [-1, -1]]).astype(np.float32)
    # STEP_SIZE = 0.015

    # for ax in ALL_AXES:
    #     # apply actions (T x B x 5)
    #     fake_actions = torch.zeros_like(data["actions"])
    #     fake_actions[:, :, 2] = data["actions"][:, :, 2]  # use same z
    #     fake_actions[:,:, :2] = torch.from_numpy(STEP_SIZE * ax)
    #     gif_name = f"diag{int(ax[0])}{int(ax[1])}_{int(STEP_SIZE * 1000)}mm.gif"
    #     plot_rollout(model, data, fake_actions, gif_name)