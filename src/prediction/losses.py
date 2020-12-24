"""Loss functions for the video prediction"""
import numpy as np
from numpy.linalg import norm
import torch
import torch.nn as nn
from skimage.filters import gaussian
from torchvision.transforms import ToTensor
from src.utils.state import State

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

# def img_l2_dist(curr_img, goal_img):
#     # makes sure to cast uint8 img to float before norming
#     dist = norm(curr_img.astype(np.float) - goal_img.astype(np.float))
#     return dist

# def eef_inpaint_cost(curr_eef, goal_eef, curr_img, goal_img, robot_weight, print_cost=False):
#     """
#     Assumes the images are inpainted.
#     """
#     eef_loss = -norm(curr_eef - goal_eef)
#     # TODO: add option for don't-care cost instead of inpaint image cost.
#     image_cost = -img_l2_dist(curr_img, goal_img)
#     if print_cost:
#         print(f"eef_cost: {robot_weight * eef_loss :.2f},  img cost: {image_cost :.2f}")
#     return robot_weight * eef_loss + image_cost

class Cost:
    """Generic Cost fn interface"""
    def __init__(self, config):
        self._config = config

    def __call__(self, curr: State, goal: State):
        raise NotImplementedError()

class RobotL2Cost(Cost):
    name="robot_l2"
    def call(self, curr_robot, goal_robot):
        if curr_robot is None or goal_robot is None:
            return 0
        # TODO: check for tensor vs np array
        return -norm(curr_robot - goal_robot)

    def __call__(self, curr: State, goal: State):
        return self.call(curr.robot, goal.robot)


class ImgL2Cost(Cost):
    name = "img_l2"
    def call(self, curr_img, goal_img):
        if curr_img is None or goal_img is None:
            return 0
        # TODO: check for tensor vs np array
        curr_img = curr_img.astype(np.float)
        goal_img = goal_img.astype(np.float)
        threshold = self._config.img_cost_threshold
        if threshold is None:
            return -norm(curr_img - goal_img)
        diff = np.abs(curr_img - goal_img)
        return -np.sum(diff > threshold)

    def __call__(self, curr: State, goal: State):
        return self.call(curr.img, goal.img)

class ImgDontcareCost(Cost):
    name = "img_dontcare"
    def call(self, curr_img, goal_img, curr_mask, goal_mask):
        if curr_img is None or goal_img is None:
            return 0
        # TODO: check for tensor vs np array
        curr_img = curr_img.astype(np.float)
        goal_img = goal_img.astype(np.float)

        total_mask = curr_mask | goal_mask
        non_robot_region1 = curr_img[~total_mask]
        non_robot_region2 = goal_img[~total_mask]
        threshold = self._config.img_cost_threshold
        if threshold is None:
            return -norm(non_robot_region1 - non_robot_region2)
        non_robot_diff = np.abs(non_robot_region1 - non_robot_region2)
        non_robot_loss = np.sum(non_robot_diff > threshold)

        if self._config.img_cost_mask_norm:
            non_robot_loss /= np.sum(~total_mask)
        return -non_robot_loss

    def __call__(self, curr: State, goal: State):
        return self.call(curr.img, goal.img, curr.mask, goal.mask)


class RobotWorldCost(Cost):
    """Combination of a robot and a world cost."""
    def __init__(self, config) -> None:
        self._config = config
        self._build_robot_cost(config)
        self._build_world_cost(config)

    def _build_robot_cost(self, config):
        self.robot_cost_weight = config.robot_cost_weight
        self.robot_cost = RobotL2Cost(config)

    def _build_world_cost(self, config):
        self.world_cost_weight = config.world_cost_weight
        self.world_cost = ImgL2Cost(config)
        if config.reward_type == "dontcare":
            self.world_cost = ImgDontcareCost(config)

    def __call__(self, curr: State, goal: State, print_cost=False):
        weights = [self.robot_cost_weight, self.world_cost_weight]
        costs = [self.robot_cost, self.world_cost]
        total_cost = 0
        for w, c in zip(weights, costs):
            if w == 0:
                continue
            cost = w * c(curr, goal)
            if print_cost:
                print(f"cost {c.name}: {cost:.2f}")
            total_cost += cost

        return total_cost
