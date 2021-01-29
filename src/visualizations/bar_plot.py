from typing import List
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pickle

"""
Create a grouped bar plot to compare methods' goal error across demonstrations
"""
def load_stats(path, num):
    with open(path, "rb") as f:
        stats = pickle.load(f)
        names = []
        demo_names = stats["demo_name"][:num]
        for name in demo_names:
            # print(name)
            n = "_".join(name.split(".")[0].split("_")[-1:])
            names.append(n)
        final_obj_dist = stats["final_obj_dist"][:num]
    name_dist = zip(names, final_obj_dist)
    return sorted(name_dist, key=lambda x: x[0])

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())
    ax.set_xlabel("Demonstration ID")
    ax.set_ylabel("Goal Error (cm)")
    ax.set_title("Goal Error Per Demonstration")

def load_all_stats(data):
    """Loads all stat files into a list
    Data is a dictionary, keyed by method name and value is pkl file path
    The list is sorted in descending order of the first methods's error value
    """
    names = list(data.keys())
    all_name_dist = defaultdict(list)
    for p in data.values():
        name_dist = load_stats(p, 100)
        for (n, d) in name_dist:
            all_name_dist[n].append(d * 100)
    stats_list = list(all_name_dist.values())
    # sorted list is [ [a, b, c, d]... ] where is a, b, c are the error values
    # for the respective methods
    stats_list.sort(key=lambda x: x[0], reverse=True)

    # filter the sorted list indexed by method
    out = defaultdict(list)
    for stats in stats_list:
        for i, s in enumerate(stats):
            name = names[i]
            out[name].append(s)
    return out

if __name__ == "__main__":
    methods = {
        "noinpaint": "noip_stats.pkl",
        # "inpaint": "inpaint_stats.pkl",
        "dontcare": "dc_stats.pkl"
    }
    success_threshold = 3.0 # in cm

    num_demos = 100
    # main("inpaint", "noinpaint", path_1, path_2, num_demos)

    # store data in list of tuples [a, b, c, d]
    # where a is the baseline error, b, c, d are the other methods errors
    # sort the tuples by value of a
    data = load_all_stats(methods) # |p| x 100 
    # measure success rate
    for method, errors in data.items():
        method_errors = np.asarray(errors)
        num_success = np.sum(method_errors <= success_threshold)
        success_percent = num_success / len(method_errors)
        print(f"{method} success: {success_percent:.2f}")
    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=.8, single_width=.9)
    fig.set_figheight(2.5)
    fig.set_figwidth(5)
    plt.margins(x=0)
    # plt.hlines(success_threshold, 0, num_demos, colors="black", linestyles="dashed")
    plt.savefig("comparison.pdf")
