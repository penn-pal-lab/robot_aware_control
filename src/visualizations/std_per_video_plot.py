import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_xy():
    # 1 robot to Franka
    # method1_metrics_path = "ckpt_eval/fulllocobot_roboaware_aug_franka/metrics.pkl"
    # method2_metrics_path = "ckpt_eval/fulllocobot_vanillastate_aug_franka/metrics.pkl"

    # multirobot to Franka
    # method1_metrics_path = "ckpt_eval/0shotlb_roboaware_franka/metrics.pkl"
    # method2_metrics_path = "ckpt_eval/0shotlb_vanillastate_franka/metrics.pkl"

    # multirobot? to modified
    method1_metrics_path = "ckpt_eval/0shotlb_roboaware_modified/metrics.pkl"
    method2_metrics_path = "ckpt_eval/0shotlb_vanillastate_modified/metrics.pkl"
    with open(method1_metrics_path, "rb") as f:
        m1_metrics = pickle.load(f)
    with open(method2_metrics_path, "rb") as f:
        m2_metrics = pickle.load(f)

    METRIC = "ssim"
    # plot_name = f"franka_0shot_multirobot_{METRIC}.pdf"
    # plot_name = f"franka_0shot_1robot_{METRIC}.pdf"
    plot_name = f"modified_0shot_multirobot_{METRIC}.pdf"

    m1 = m1_metrics[METRIC]
    m2 = m2_metrics[METRIC]
    fig, ax = plt.subplots()
    plt.scatter(x=m1, y=m2, marker=".")
    # draw diag line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.5, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.xlabel(f"RA {METRIC.upper()}")
    plt.ylabel(f"VF+State {METRIC.upper()}")
    plt.title(f"{METRIC.upper()} Error per Video of VFS and RA")
    plt.savefig(plot_name)

if __name__ == "__main__":
    plot_xy()