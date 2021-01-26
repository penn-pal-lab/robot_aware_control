import cv2
import numpy as np
from tqdm import tqdm
import os
import datetime


tip_coord = []
SCALE = 4       # how much larger to display the image


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global tip_coord
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        tip_coord = [x, y]


def annotate_img(img):
    go_back = False
    is_fail = False
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", img[:, :, ::-1])
        key = cv2.waitKey(1) & 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == 32:   # space
            break
        elif key == ord("g"):
            is_fail = False
        elif key == ord("f"):
            is_fail = True
        elif key == ord("r"):
            go_back = True
            break
    cv2.destroyAllWindows()
    return go_back, is_fail


def display_annotation(img, labels):
    cv2.namedWindow("image")
    scaled_x = int(labels[0, 1] * SCALE)
    scaled_y = int(labels[0, 0] * SCALE)
    img[scaled_x - 3:scaled_x + 3, scaled_y - 3:scaled_y + 3] = [1.0, 0.0, 0.0]
    cv2.imshow("image", img)
    key = cv2.waitKey(0) & 0xFF   # half a second
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread("images/exp_0_img_0.png")
    print(img.shape)
    img = cv2.resize(img, (img.shape[1] * SCALE, img.shape[0] * SCALE))

    labels = np.empty((1, 2))

    go_back, is_fail = annotate_img(img)
    labels[0, 0] = tip_coord[0] / SCALE
    labels[0, 1] = tip_coord[1] / SCALE
    print(labels)

    display_annotation(img, labels)
    print("Congrats, you're done with this one!")
