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

def main(method_1, method_2, path_1, path_2, num_demos):
    stats_1 = load_stats(path_1, num_demos)
    stats_2 = load_stats(path_2, num_demos)
    labels = []
    values_1 = []
    values_2 = []
    for s1, s2 in zip(stats_1, stats_2):
        assert s1[0] == s2[0]
        labels.append(s1[0])
        values_1.append(round(s1[1] * 100, 2))
        values_2.append(round(s2[1] * 100, 2))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, values_1, width, label=method_1)
    rects2 = ax.bar(x + width / 2, values_2, width, label=method_2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Goal Error (cm)")
    ax.set_title("Goal Error Per Demonstration")
    ax.set_xticks(x)
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 10 != 0:
            label.set_visible(False)
    # ax.set_xticklabels(labels)
    ax.set_xlabel("Demonstration ID")
    ax.tick_params(axis='x', labelsize=8)
    ax.legend()


    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate(
    #             "{}".format(height),
    #             xy=(rect.get_x() + rect.get_width() / 2, height),
    #             xytext=(0, 3),  # 3 points vertical offset
    #             textcoords="offset points",
    #             ha="center",
    #             va="bottom",
    #         )


    # autolabel(rects1)
    # autolabel(rects2)

    # fig.tight_layout()
    fig.set_figheight(5)
    fig.set_figwidth(20)
    plt.savefig("comparison.pdf")

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
    fig.set_figheight(5)
    fig.set_figwidth(20)
    ax.set_xlabel("Demonstration ID")
    ax.set_ylabel("Goal Error (cm)")
    ax.set_title("Goal Error Per Demonstration")
    plt.savefig("comparison.pdf")

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
        "noinpaint": "noinpaint_real_stats.pkl",
        "inpaint": "inpaint_real_stats.pkl",
        "dontcare": "dontcare_real_stats.pkl"
    }

    # num_demos = 100
    # main("inpaint", "noinpaint", path_1, path_2, num_demos)

    # store data in list of tuples [a, b, c, d]
    # where a is the baseline error, b, c, d are the other methods errors
    # sort the tuples by value of a
    data = load_all_stats(methods) # |p| x 100 
    fig, ax = plt.subplots()
    bar_plot(ax, data, total_width=.8, single_width=.9)