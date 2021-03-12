import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def get_dim_inds(generalized_tensor):
    """ Returns a tuple 0..length, where length is the number of dimensions of the tensors"""
    return tuple(range(len(generalized_tensor.shape)))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


@torch.no_grad()
def ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel).detach().cpu().numpy()

@torch.no_grad()
def psnr(estimates, targets, data_dims=3):
    # NOTE: PSNR is not dimension-independent. The number of dimensions which are part of the metric has to be specified
    # I.e 2 for grayscale, 3 for color images.
    # estimates = estimates.cpu().numpy()
    # targets = targets.cpu().numpy()
    estimates = (estimates + 1) / 2
    targets = (targets + 1)/2

    max_pix_val = 1.0
    tolerance = 0.01
    # assert (0 - tolerance) <= np.min(targets) and np.max(targets) <= max_pix_val * (1 + tolerance)
    # assert (0 - tolerance) <= np.min(estimates) and np.max(estimates) <= max_pix_val * (1 + tolerance)
    if not(0 - tolerance) <= targets.min() and targets.max() <= max_pix_val * (1 + tolerance):
        import ipdb; ipdb.set_trace()

    if not (0 - tolerance) <= estimates.min() and estimates.max() <= max_pix_val * (1 + tolerance):
        import ipdb; ipdb.set_trace()

    mse = (estimates - targets).pow(2)
    mse_mean = mse.mean(dim=get_dim_inds(mse)[-data_dims:])
    psnr = 10 * (max_pix_val/mse_mean).log() / np.log(10)

    return psnr