"""Loss functions for the video prediction"""
import numpy as np
from numpy.linalg import norm
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
        img2 = img[self.img_dim :]
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

def img_l2_dist(curr_img, goal_img):
    # makes sure to cast uint8 img to float before norming
    dist = norm(curr_img.astype(np.float) - goal_img.astype(np.float))
    return dist

def eef_inpaint_cost(curr_eef, goal_eef, curr_img, goal_img, robot_weight, print_cost=False):
    """
    Assumes the images are inpainted.
    """
    eef_loss = -norm(curr_eef - goal_eef)
    # TODO: add option for don't-care cost instead of inpaint image cost.
    image_cost = -img_l2_dist(curr_img, goal_img)
    if print_cost:
        print(f"eef_cost: {robot_weight * eef_loss :.2f},  img cost: {image_cost :.2f}")
    return robot_weight * eef_loss + image_cost