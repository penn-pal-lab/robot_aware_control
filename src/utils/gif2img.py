from time import time
import numpy as np
import imageio
import os
from PIL import Image, ImageSequence
import cv2
from skimage.color import rgba2rgb
import ipdb


def gif2imgs(path, size=(48, 64), pad=0):
    gif = imageio.get_reader(path + ".gif")
    gif_shape = None
    for t, frame in enumerate(gif):
        print(frame.shape)  # Each frame is a numpy matrix
        gif_shape = frame.shape
        # assert frame.shape[0] % size[0] == 0 and frame.shape[1] % size[1] == 0
        # frame = np.asarray(frame)
        # frame = rgba2rgb(frame)
        # # imageio.imwrite(path + "_" + str(t) + ".png", frame[:,:,:3])
        # cv2.imwrite(path + "_" + str(t) + ".png", frame[:,:,:3])

    for i in range(gif_shape[0] // size[0]):
        for j in range(gif_shape[1] // size[1]):
            for t, frame in enumerate(gif):
                row_col_dir = "/" + str(i) + "_" + str(j)
                time_dir = "/t=" + str(t)
                os.makedirs(path + row_col_dir, exist_ok=True)

                img = frame[
                    i * size[0] : (i + 1) * size[0],
                    j * size[1] : (j + 1) * size[1],
                    :,
                ]

                img = np.copy(img[:, :, :3], order="F")
                # imageio.imwrite(
                #     path + time_dir + "/subimg_" + str(i) + "_" + str(j) + ".png", img
                # )
                # imageio.imwrite(
                #     path + row_col_dir + "/subimg_" + str(t) + ".png", img
                # )
                im = Image.fromarray(img)
                im.save(path + row_col_dir + "/subimg_" + str(t) + ".png")
                # imageio.imwrite(path + "_" + str(t) + ".png", frame)


def slice_image(img, t, parent_path, size=(48, 64), pad=0):
    # img = imageio.imread(path)
    img_shape = img.shape
    for i in range(img_shape[0] // size[0]):
        for j in range(img_shape[1] // size[1]):
            row_col_dir = "/" + str(i) + "_" + str(j)
            os.makedirs(parent_path + row_col_dir, exist_ok=True)

            subimg = img[
                i * size[0] : (i + 1) * size[0],
                j * size[1] + pad * j: (j + 1) * size[1] + pad * j,
                :,
            ]

            imageio.imwrite(parent_path + row_col_dir + "/subimg_" + str(t) + ".png", subimg)


if __name__ == "__main__":
    GIF_PATH = "/home/pallab/locobot_ws/src/roboaware/figures/rollout_vis/fewshot_locobot_v_row7"
    # GIF_PATH = "/home/pallab/locobot_ws/src/roboaware/figures/rollout_vis/sim_0shotfetch_ra_row4"
    # GIF_PATH = "/home/pallab/locobot_ws/src/roboaware/figures/test_for_gif2imgs"

    # gif2imgs(GIF_PATH, pad=1)
    for t in range(1,7):
        img = imageio.imread(GIF_PATH + "_" + str(t) + ".png")
        slice_image(img, t, GIF_PATH, pad=1)