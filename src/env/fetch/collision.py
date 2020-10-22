from abc import ABC, abstractmethod

import numpy as np


class CollisionObject(ABC):
    """
    Abstract class for a parametrically defined collision object.
    """
    @abstractmethod
    def in_collision(self, target):
        """
        Checks whether target point is in collision. Points at the boundary of
        the object are in collision.

        :returns: Boolean indicating target is in collision.
        """
        pass


class CollisionBox(CollisionObject):
    """
    N-dimensional box collision object.
    """
    def __init__(self, location, half_lengths):
        """
        :params location: coordinates of the center
        :params half_lengths: half-lengths of the rectangle along each axis
        """
        self.location = np.asarray(location)
        self.half_lengths = np.asarray(half_lengths)
        self.ndim = self.location.shape[0]

    def in_collision(self, target):
        # the point must be within all dimensions of the rectangle
        for t, x, half in zip(target, self.location, self.half_lengths):
            low = x - half
            high = x +  half
            if not (low <= t <= high):
                return False
        return True


class CollisionSphere(CollisionObject):
    """
    N-dimensional sphere collision object.
    """
    def __init__(self, location, radius):
        """
        :params location: coordinates of the center
        :params radius: radius of the circle
        """
        self.location = np.asarray(location)
        self.radius = radius

    def in_collision(self, target):
        return np.linalg.norm(target - self.location) <= self.radius

    def line_in_collision(self, o, u):
        """
        https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        Check if a line intersects with this sphere
        o = starting point of line
        u = direction of line
        """
        c = self.location
        r = self.radius
        delta = (np.dot(u, (o - c)))**2 - (np.linalg.norm(o-c)**2 - r**2)
        return delta >= 0

