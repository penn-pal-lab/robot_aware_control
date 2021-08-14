import torch
import numpy as np
from torch import Tensor

def zero_robot_region(mask, image, inplace=False):
    """
    Set the robot region to zero
    """
    if isinstance(mask, Tensor):
        robot_mask = mask.type(torch.bool)
        robot_mask = robot_mask.repeat(1, 3, 1, 1)
        if not inplace:
            image = image.clone()
        image[robot_mask] *= 0
    else:
        robot_mask = mask.astype(bool)
        if not inplace:
            image = image.copy()
        image[robot_mask] = 0
    return image