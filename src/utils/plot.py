import os

import imageio
import ipdb
import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
from PIL import Image
import cv2
from functools import partial

putText = partial(
    cv2.putText,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.3,
    color=(0, 0, 0),
    thickness=1,
    lineType=cv2.LINE_AA,
)


def is_sequence(arg):
    return (
        not hasattr(arg, "strip")
        and not type(arg) is np.ndarray
        and not hasattr(arg, "dot")
        and (hasattr(arg, "__getitem__") or hasattr(arg, "__iter__"))
    )


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(
            c_dim, x_dim * len(images) + padding * (len(images) - 1), y_dim
        )
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding : (i + 1) * x_dim + i * padding, :].copy_(
                image
            )

        return result

    # if this is just a list, make a stacked image
    else:
        images = [
            x.data if isinstance(x, torch.autograd.Variable) else x for x in inputs
        ]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(
            c_dim, x_dim, y_dim * len(images) + padding * (len(images) - 1)
        )
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding : (i + 1) * y_dim + i * padding].copy_(
                image
            )
        return result


def save_np_img(fname, x):
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    img = Image.fromarray(x, high=255 * x.max(), channel_axis=0)
    img.save(fname)


def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    img = (255 * tensor.permute(1, 2, 0).numpy()).astype(np.uint8)
    return Image.fromarray(img)


def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x * 255))
    draw = Image.Draw(pil)
    draw.text((4, 64), text, (0, 0, 0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.0)).transpose(1, 2).transpose(0, 1)


def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu().transpose_(0, 1).transpose_(1, 2).clamp_(0, 1).numpy()
        images.append(np.uint8(img * 255))
    imageio.mimsave(filename, images, duration=duration)


def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor(
            [draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0
        )
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1).numpy()
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)


def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)
