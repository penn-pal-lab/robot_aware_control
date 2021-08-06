import torch

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