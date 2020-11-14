import matplotlib

matplotlib.use("agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.env.fetch.collision import CollisionBox, CollisionSphere
from src.env.fetch.rrt import RRT


class PlanarRRT(RRT):
    """
    Implementation of Rapidly-Exploring Random Trees (RRT) for 2D search spaces.
    """

    def __init__(
        self,
        visualize=True,
        visualize_every=5,
        visualize_path="test.png",
        *args,
        **kwargs
    ):
        """
        :param visualize: boolean flag for whether to visualize RRT as it
            builds.
        :param visualize_every: How many build steps between render frames.
        """
        super(PlanarRRT, self).__init__(*args, **kwargs)

        self.visualize = visualize
        self.visualize_every = visualize_every
        self.visualize_path = visualize_path
        if self.visualize:
            self.fignum = plt.figure().number

    def build(self):
        """
        Build an RRT with a visualization.

        In each step of the RRT:
            1. Sample a random point.
            2. Find its nearest neighbor.
            3. Attempt to create a new node in the direction of sample from its
                nearest neighbor.
            4. If we have created a new node, check for completion.

        Once the RRT is complete, add the goal node to the RRT and build a path
        from start to goal.

        :returns: A list of states that create a path from start to
            goal on success. On failure, returns None.
        """
        for k in range(self.max_iter):
            r = self._get_random_sample()
            neighbor = self._get_nearest_neighbor(r)
            new_node = self._extend_sample(r, neighbor)
            if new_node and self._check_for_completion(new_node):
                self.goal.parent = new_node
                new_node.children.append(self.goal)
                path = self._trace_path_from_start(self.goal)
                self._visualize(path=path, hold=True)
                return path
            elif k % self.visualize_every == 0:
                self._visualize()

        print(
            "Failed to find path from {0} to {1} after {2} iterations!".format(
                self.start.state, self.goal.state, self.max_iter
            )
        )
        return None

    def _visualize(self, path=None, hold=False):
        if not self.visualize:
            return
        fig = plt.figure(self.fignum)
        ax = plt.gca()
        plt.cla()
        for node in self.start:
            if node.parent:
                plt.plot(
                    (node.state[0], node.parent.state[0]),
                    (node.state[1], node.parent.state[1]),
                    "-b",
                )

        for obs in self.obstacles:
            if isinstance(obs, CollisionBox):
                p = patches.Rectangle(
                    obs.location - obs.half_lengths,
                    obs.half_lengths[0] * 2,
                    obs.half_lengths[1] * 2,
                    color="k",
                )
            elif isinstance(obs, CollisionSphere):
                p = patches.Circle(obs.location, obs.radius, color="k")
            ax.add_patch(p)

        plt.plot(self.start.state[0], self.start.state[1], "*g")
        plt.plot(self.goal.state[0], self.goal.state[1], "*r")
        ax.set_xlim(self.dim_ranges[0])
        ax.set_ylim(self.dim_ranges[1])

        if path is not None:
            for i in range(1, len(path)):
                plt.plot(
                    (path[i][0], path[i - 1][0]), (path[i][1], path[i - 1][1]), "-r"
                )

        # if hold:
        #    plt.show()
        # else:
        #    plt.pause(0.01)
        if hold:
            plt.axes().set_aspect("equal")
            plt.savefig(self.visualize_path)


if __name__ == "__main__":
    rrt = PlanarRRT(
        start_state=[0.2, 0.2], goal_state=[0.7, 0.7], dim_ranges=[(0, 1), (0, 1)]
    )
    path = rrt.build()
    print(path)
