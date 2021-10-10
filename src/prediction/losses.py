"""Loss functions for the video prediction"""
import numpy as np
from numpy.linalg import norm
import torch
from torch.functional import Tensor
import torch.nn as nn
from skimage.filters import gaussian
from torchvision.transforms import ToTensor
from src.utils.state import State

mse_criterion = nn.MSELoss()

def l1_criterion(prediction, target, batch_weight=None):
    diff = target - prediction # B, 3, H, W
    if batch_weight is None:
        mean_err = (diff.abs_()).mean()
    else:
        mean_err = torch.mean(batch_weight * (diff.abs_()).mean((1,2,3)))
    return mean_err

def dontcare_mse_criterion(prediction, target, mask, robot_weight):
    """
    Zero out the robot region from the target image before summing up the cost
    prediction / target is B x C x H x W
    mask is B x 1 x H x W
    """
    diff = target - prediction # 3 x H x W
    mask = mask.type(torch.bool)
    repeat_mask = mask.repeat(1,3,1,1) # repeat channel dim
    diff[repeat_mask] *= robot_weight
    num_world_pixels = (~repeat_mask).sum((1,2,3)) + 1
    mean_err = torch.mean((diff ** 2).sum((1,2,3)) / num_world_pixels)
    return mean_err

def dontcare_l1_criterion(prediction, target, mask, robot_weight, batch_weight=None):
    """
    Zero out the robot region from the target image before summing up the cost
    prediction / target is B x C x H x W
    mask is B x 1 x H x W
    """
    diff = target - prediction # B, 3, H, W
    mask = mask.type(torch.bool)
    repeat_mask = mask.repeat(1,3,1,1) # repeat channel dim
    diff[repeat_mask] *= robot_weight
    num_world_pixels = (~repeat_mask).sum((1,2,3)) + 1
    if batch_weight is None:
        mean_err = torch.mean((diff.abs_()).sum((1,2,3)) / num_world_pixels)
    else:
        mean_err = torch.mean(batch_weight * (diff.abs_()).sum((1,2,3)) / num_world_pixels)
    return mean_err

def robot_mse_criterion(prediction, target, mask):
    """
    MSE of the robot pixels
    prediction / target is B x C x H x W
    mask is B x 1 x H x W
    """
    diff = target - prediction # 3 x H x W
    mask = mask.type(torch.bool)
    repeat_mask = mask.repeat(1,3,1,1) # repeat channel dim
    diff[~repeat_mask] = 0
    num_robot_pixels = repeat_mask.sum((1,2,3)) + 1
    mean_err = torch.mean((diff ** 2).sum((1,2,3)) / num_robot_pixels)
    return mean_err

def world_mse_criterion(prediction, target, mask):
    """
    MSE of the world pixels
    prediction / target is B x C x H x W
    mask is B x 1 x H x W
    """
    diff = target - prediction # 3 x H x W
    mask = mask.type(torch.bool)
    repeat_mask = mask.repeat(1,3,1,1) # repeat channel dim
    diff[repeat_mask] = 0
    num_world_pixels = (~repeat_mask).sum((1,2,3)) + 1
    mean_err = torch.mean((diff ** 2).sum((1,2,3)) / num_world_pixels)
    return mean_err

def world_psnr_criterion(prediction, target, mask):
    """
    MSE of the world pixels
    prediction / target is B x C x H x W
    mask is B x 1 x H x W
    """
    diff = target - prediction # 3 x H x W
    mask = mask.type(torch.bool)
    repeat_mask = mask.repeat(1,3,1,1) # repeat channel dim
    diff[repeat_mask] = 0
    num_world_pixels = (~repeat_mask).sum((1,2,3)) + 1
    batch_mse = (diff ** 2).sum((1,2,3)) / num_world_pixels
    psnr = 10 * (1 / batch_mse).log() / np.log(10)
    # mean_err = torch.mean(batch_mse)
    return psnr


def kl_criterion(mu1, logvar1, mu2, logvar2, bs):
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = (
        torch.log(sigma2 / sigma1)
        + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2))
        - 1 / 2
    )
    assert kld.shape[0] == bs, f"{kld.shape[0]} != {bs}"
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
    name = "generic_cost"
    def __init__(self, config):
        self._config = config

    def __call__(self, curr: State, goal: State):
        raise NotImplementedError()

class RobotL2Cost(Cost):
    name="robot_l2"
    def _call(self, curr_robot, goal_robot):
        if curr_robot is None or goal_robot is None:
            return 0.0
        return -norm(curr_robot - goal_robot)

    def _call_tensor(self, curr_robot, goal_robot):
        if curr_robot is None or goal_robot is None:
            return 0.0
        state_diff = (curr_robot - goal_robot) ** 2
        if len(state_diff.shape) == 2:
            # batch vector version
            sum_diff = torch.sum(state_diff, (1))
        elif len(state_diff.shape) == 1:
            # single vector version
            sum_diff = torch.sum(state_diff)
        else:
            raise NotImplementedError(f"Tensor shape {state_diff.shape} not supported")
        dist = sum_diff.sqrt().cpu().numpy()
        return -dist

    def __call__(self, curr: State, goal: State):
        if isinstance(curr.state, Tensor) or isinstance(goal.state, Tensor):
            return self._call_tensor(curr.state, goal.state)
        return self._call(curr.state, goal.state)


class ImgL2Cost(Cost):
    name = "img_l2"
    def _call(self, curr_img, goal_img):
        if curr_img is None or goal_img is None:
            return 0
        curr_img = curr_img.astype(np.float)
        goal_img = goal_img.astype(np.float)
        threshold = self._config.img_cost_threshold
        if threshold is None:
            dist = norm(curr_img - goal_img)
        else:
            diff = np.abs(curr_img - goal_img)
            dist = np.sum(diff > threshold)
        return -dist

    def _call_tensor(self, curr_img: Tensor, goal_img: Tensor):
        if curr_img is None or goal_img is None:
            return 0
        img_diff = (255 * (curr_img - goal_img)) ** 2
        if len(img_diff.shape) == 4: # batch x |img|
            sum_diff = torch.sum(img_diff, (1, 2, 3)) # sum up across image dimensions
        elif len(img_diff.shape) == 3: # img only
            sum_diff = torch.sum(img_diff)
        else:
            raise NotImplementedError(f"Tensor shape {img_diff.shape} not supported")
        dist = sum_diff.sqrt().cpu().numpy()
        return -dist

    def __call__(self, curr: State, goal: State):
        if isinstance(curr.img, Tensor) or isinstance(goal.img, Tensor):
            return self._call_tensor(curr.img, goal.img)
        return self._call(curr.img, goal.img)

class ImgDontcareCost(Cost):
    name = "img_dontcare"
    def _call_tensor(self, curr_img, goal_img, curr_mask, goal_mask):
        if curr_img is None or goal_img is None:
            return 0
        curr_mask = curr_mask.type(torch.bool)
        goal_mask = goal_mask.type(torch.bool)
        img_diff = (255 * (curr_img - goal_img)) ** 2
        total_mask_2d = curr_mask | goal_mask
        total_mask = total_mask_2d.repeat(1,3,1,1)
        img_diff[total_mask] = 0 # set robot region to 0
        if len(img_diff.shape) == 4: # batch x |img|
            sum_diff = torch.sum(img_diff, (1, 2, 3)) # sum up across image dimensions
        elif len(img_diff.shape) == 3: # img only
            sum_diff = torch.sum(img_diff)
        else:
            raise NotImplementedError(f"Tensor shape {img_diff.shape} not supported")
        dist = sum_diff.sqrt()
        num_world_pixels = torch.sum(~total_mask_2d, (1,2,3))
        dist /= num_world_pixels
        dist = dist.cpu().numpy()
        return -dist

    def _call(self, curr_img, goal_img, curr_mask, goal_mask):
        if curr_img is None or goal_img is None:
            return 0
        curr_img = curr_img.astype(np.float)
        goal_img = goal_img.astype(np.float)

        total_mask = curr_mask | goal_mask
        non_robot_region1 = curr_img[~total_mask]
        non_robot_region2 = goal_img[~total_mask]
        threshold = self._config.img_cost_threshold
        if threshold is None:
            non_robot_loss = norm(non_robot_region1 - non_robot_region2)
        else:
            non_robot_diff = np.abs(non_robot_region1 - non_robot_region2)
            non_robot_loss = np.sum(non_robot_diff > threshold)

        if self._config.img_cost_world_norm:
            non_robot_loss /= np.sum(~total_mask)
        return -non_robot_loss

    def __call__(self, curr: State, goal: State):
        if isinstance(curr.img, Tensor) or isinstance(goal.img, Tensor): return self._call_tensor(curr.img, goal.img, curr.mask, goal.mask)
        return self._call(curr.img, goal.img, curr.mask, goal.mask)


class RobotWorldCost(Cost):
    """Combination of a robot and a world cost."""
    def __init__(self, config):
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

    def __call__(self, curr: State, goal: State, print_cost=False, return_info=False):
        weights = [self.robot_cost_weight, self.world_cost_weight]
        costs = [self.robot_cost, self.world_cost]
        total_cost = 0
        print_str = ""
        info = {}
        for w, c in zip(weights, costs):
            if w == 0:
                continue
            cost = w * c(curr, goal)
            if return_info:
                if type(cost) in [np.float64, float]:
                    info[c.name] = cost
                else:
                    # batched version
                    raise NotImplementedError()
            if print_cost:
                if type(cost) in [np.float64, float]:
                    print_str += f"{c.name}: {cost:.4f} ,"
                else:
                    for cost_val in cost:
                        print_str += f" {c.name}: {cost_val:.4f} ,"
            total_cost += cost
        if print_cost:
            print(print_str)
        if return_info:
            return total_cost, info

        return total_cost
