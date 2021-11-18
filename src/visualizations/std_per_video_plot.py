import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_xy(m1_path, m2_path, metric, plot_name):
    # 1 robot to Franka
    # method1_metrics_path = "ckpt_eval/fulllocobot_roboaware_aug_franka/metrics.pkl"
    # method2_metrics_path = "ckpt_eval/fulllocobot_vanillastate_aug_franka/metrics.pkl"

    # multirobot to Franka
    # method1_metrics_path = "ckpt_eval/0shotlb_roboaware_franka/metrics.pkl"
    # method2_metrics_path = "ckpt_eval/0shotlb_vanillastate_franka/metrics.pkl"

    # multirobot? to modified
    # method1_metrics_path = "ckpt_eval/0shotlb_roboaware_modified/metrics.pkl"
    # method2_metrics_path = "ckpt_eval/0shotlb_vanillastate_modified/metrics.pkl"
    with open(m1_path, "rb") as f:
        m1_metrics = pickle.load(f)
    with open(m2_path, "rb") as f:
        m2_metrics = pickle.load(f)

    m1 = m1_metrics[metric]
    m2 = m2_metrics[metric]
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

    plt.xlabel(f"RA {metric.upper()}")
    plt.ylabel(f"VF+State {metric.upper()}")
    plt.title(f"{metric.upper()} Error per Video of VFS and RA")
    plt.savefig(plot_name + ".pdf")
    plt.savefig(plot_name + ".png", dpi=400)
    plt.savefig(plot_name)

    # count how many times m1 is better than m2 for the metric
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    m1_better = np.sum(m1 > m2)
    total = len(m1)
    print(metric, "m1 is better", f"{m1_better} / {total}")

if __name__ == "__main__":
    print("1 robot to Franka")
    m1_path = "ckpt_eval/fulllocobot_roboaware_aug_franka/metrics.pkl"
    m2_path = "ckpt_eval/fulllocobot_vanillastate_aug_franka/metrics.pkl"
    metric="psnr"
    plot_name = f"franka_0shot_1robot_{metric}"
    plot_xy(m1_path, m2_path, metric, plot_name)
    metric="ssim"
    plot_name = f"franka_0shot_1robot_{metric}"
    plot_xy(m1_path, m2_path, metric, plot_name)


    print("multirobot to Franka")
    m1_path = "ckpt_eval/0shotlb_roboaware_franka/metrics.pkl"
    m2_path = "ckpt_eval/0shotlb_vanillastate_franka/metrics.pkl"
    metric="psnr"
    plot_name = f"franka_0shot_multirobot_{metric}"
    plot_xy(m1_path, m2_path, metric, plot_name)
    metric="ssim"
    plot_name = f"franka_0shot_multirobot_{metric}"
    plot_xy(m1_path, m2_path, metric, plot_name)

    print("multirobot to Modified WidowX")
    m1_path = "ckpt_eval/0shotlb_roboaware_modified/metrics.pkl"
    m2_path = "ckpt_eval/0shotlb_vanillastate_modified/metrics.pkl"
    metric="psnr"
    plot_name = f"modified_0shot_multirobot_{metric}"
    plot_xy(m1_path, m2_path, metric, plot_name)
    metric="ssim"
    plot_name = f"modified_0shot_multirobot_{metric}"
    plot_xy(m1_path, m2_path, metric, plot_name)
