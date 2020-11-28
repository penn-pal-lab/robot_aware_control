"""Loss functions for the video prediction"""
import numpy as np
import torch
import torch.nn as nn
from skimage.filters import gaussian
from torchvision.transforms import ToTensor

mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()


def kl_criterion(mu1, logvar1, mu2, logvar2, bs):
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = (
        torch.log(sigma2 / sigma1)
        + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2))
        - 1 / 2
    )
    assert kld.shape[0] == bs
    return kld.sum() / bs


class InpaintBlurCost:
    def __init__(self, config) -> None:
        self.blur_width = config.img_dim * 2
        self.sigma = config.blur_sigma
        self.unblur_cost_scale = config.unblur_cost_scale
        self.img_dim = config.img_dim
        self.blur_img = self._blur_single
        self.to_tensor = ToTensor()
        if config.multiview:
            self.blur_img = self._blur_multiview

    def _blur(self, img):
        s = self.sigma
        w = self.blur_width
        t = (((w - 1) / 2) - 0.5) / s
        blur = (255 * gaussian(img, sigma=s, truncate=t, multichannel=True)).astype(
            np.uint8
        )
        return blur

    def _blur_single(self, img):
        img = img.cpu().permute(1, 2, 0)
        return self._blur(img)

    def _blur_multiview(self, img):
        img = img.cpu().permute(1, 2, 0)
        img1 = img[: self.img_dim]
        img2 = img[self.img_dim:]
        blur_img1 = self._blur(img1)
        blur_img2 = self._blur(img2)
        blur_img = np.concatenate([blur_img1, blur_img2])
        return blur_img

    def __call__(self, img, goal, blur=True):
        scale = -1
        if blur:
            # imageio.imwrite("img.png", img.permute(1,2,0))
            img = self.to_tensor(self.blur_img(img))
            # imageio.imwrite("blur_img.png", img.permute(1,2,0))
            # ipdb.set_trace()
            goal = self.to_tensor(self.blur_img(goal))
        else:
            scale = -1 * self.unblur_cost_scale

        cost = scale * mse_criterion(img, goal)
        return cost


def img_diff(img1, img2, thres):
    """
    img: numpy array
    """
    diff = np.abs(img1 - img2)
    return np.sum(diff > thres) / img1.size


def weighted_img_diff(img1, img2, robot_mask1, robot_mask2, robot_w=0.01, thres=3):
    """
    Inputs:
        img: numpy array
        robot_mask: numpy array of bools
        robot_w: weight for robot-region image loss, default to 0.01
    """
    total_mask = robot_mask1 | robot_mask2
    robot_region1 = img1[total_mask]
    robot_region2 = img2[total_mask]
    robot_diff = np.abs(robot_region1 - robot_region2)
    robot_loss = np.sum(robot_diff > thres) / np.sum(total_mask)  # larger than 1 because RGB

    non_robot_region1 = img1[~total_mask]
    non_robot_region2 = img2[~total_mask]
    non_robot_diff = np.abs(non_robot_region1 - non_robot_region2)
    non_robot_loss = np.sum(non_robot_diff > thres) / np.sum(~total_mask)

    # print(f"robot_loss: {robot_loss:.2f}, non_robot_loss: {non_robot_loss:.2f}")
    return robot_w * robot_loss + (1 - robot_w) * non_robot_loss


def pose_img_cost(img1, img2, robot_mask1, robot_mask2, curr_eef, goal_eef, robot_w, thres=3):
    """
    Inputs:
        img: numpy array
        robot_mask: numpy array of bools
        robot_w: weight for robot-region image loss
        curr_eef: current end-effector pose, numpy array
        goal_eef: target end-effector pose, numpy array
    """
    total_mask = robot_mask1 | robot_mask2
    robot_loss = np.linalg.norm(curr_eef - goal_eef)

    non_robot_region1 = img1[~total_mask]
    non_robot_region2 = img2[~total_mask]
    non_robot_diff = np.abs(non_robot_region1 - non_robot_region2)
    non_robot_loss = np.sum(non_robot_diff > thres) / np.sum(~total_mask)

    return robot_w * robot_loss + (1 - robot_w) * non_robot_loss
