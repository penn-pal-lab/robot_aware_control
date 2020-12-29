import matplotlib
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
            print(name)
            n = "_".join(name.split(".")[0].split("_")[-1:])
            names.append(n)
        final_obj_dist = stats["final_obj_dist"][:num]
    name_dist = zip(names, final_obj_dist)
    # return sorted(name_dist, key=lambda x: x[0])
    return name_dist

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
    ax.set_title("Goal Error Per Episode")
    ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    ax.set_xlabel("Demonstration ID")
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
    fig.set_figwidth(100)
    plt.show()

if __name__ == "__main__":
    path_1 = "dontcare_stats.pkl"
    path_2 = "threshold_dontcare_stats.pkl"
    num_demos = 30
    main("dontcare", "threshold_dontcare", path_1, path_2, num_demos)