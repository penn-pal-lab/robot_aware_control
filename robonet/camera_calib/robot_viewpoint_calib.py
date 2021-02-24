import cv2
import numpy as np
import os
import argparse
import imageio
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import ipdb

from robonet.camera_calib.robonet_calibration import display_annotation


tip_coord = []

SCALE = 4  # how much larger to display the image
VISUAL_REPROJ = True


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
            break
        elif key == ord("r"):
            go_back = True
            break
    cv2.destroyAllWindows()
    return go_back, is_fail

def calibrate_viewpoint():
    parser = argparse.ArgumentParser(
        description="calibrate specific robot and viewpoint"
    )
    parser.add_argument("robot", type=str, help="robot")
    parser.add_argument("viewpoint", type=str, help="viewpoint")
    parser.add_argument("num_trajectories", type=int, default=3, help="number of trajectories used for calibration")
    parser.add_argument(
        "--direct_calibrate", action="store_true", help="directly calibrate if annotation was done"
    )
    parser.add_argument(
        "--visual_distribution", action="store_true", help="visualize the distribution of eef"
    )

    args = parser.parse_args()

    target_dir = args.robot + "/" + args.viewpoint
    calibrate(args, target_dir)

def calibrate_directory():
    parser = argparse.ArgumentParser(
        description="calibrate given directory"
    )
    parser.add_argument("--calibrate_dir", type=str)
    parser.add_argument(
        "--direct_calibrate", action="store_true", help="directly calibrate if annotation was done"
    )
    parser.add_argument(
        "--visual_distribution", action="store_true", help="visualize the distribution of eef"
    )

    args = parser.parse_args()

    target_dir = args.calibrate_dir
    calibrate(args, target_dir)

def calibrate(args, target_dir):
    use_for_calibration = []
    for f in os.scandir(args.calibrate_dir):
        if not f.name[-3:] == "gif":
            continue
        parts = f.name.split(".")
        name = parts[0]
        use_for_calibration.append(name)
    use_for_calibration = use_for_calibration[:1]
    print("calibrating", use_for_calibration)
    if not args.direct_calibrate:
        all_pixel_coords = []
        all_3d_pos = []
        num_annotated = 0
        # TODO: change what experiment to load
        for exp_id in use_for_calibration:
            states = np.load(target_dir + "/states_" + exp_id + ".npy")

            labels = []
            temp_states = []

            gif = imageio.get_reader(target_dir + "/" + exp_id + ".gif")
            t = 0
            for img in gif:
                img = img[:, :, :3]
                print(img.shape)
                img = cv2.resize(
                    img, (img.shape[1] * SCALE, img.shape[0] * SCALE))

                go_back, is_fail = annotate_img(img)

                if not is_fail:
                    x = tip_coord[0] / SCALE
                    y = tip_coord[1] / SCALE
                    display_annotation(img, [x, y])
                    temp_states.append(states[t])
                    labels.append([x, y])
                    print(labels[-1])
                else:
                    print("skip label")
                num_annotated += 1
                print("Annotated", num_annotated)
                t += 1

            all_pixel_coords.extend(labels)  # |exp * T| x 2
            all_3d_pos.extend(temp_states)  # |exp * T| x 3

        all_pixel_coords = np.array(all_pixel_coords)
        all_3d_pos = np.array(all_3d_pos)

        np.save(target_dir + "/all_pixel_coords", all_pixel_coords)
        np.save(target_dir + "/all_3d_pos", all_3d_pos)
        print("Congrats, you're done with this one!")
    else:
        all_pixel_coords = np.load(target_dir + "/all_pixel_coords.npy")
        all_3d_pos = np.load(target_dir + "/all_3d_pos.npy")
        print("pixel coords shape", all_pixel_coords.shape)
        print("loaded 3d pos shape", all_3d_pos.shape)

    # calibration section starts here
    all_3d_pos = np.array(all_3d_pos[:, 0:3])
    print("3d pos shape", all_3d_pos.shape)

    all_pixel_coords = np.array(all_pixel_coords, dtype=np.float32)
    intrinsic_guess = np.array([[320.75, 0, 160],
                                [0, 320.75, 120],
                                [0, 0, 1]])
    img_shape = (240, 320)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH
    # flags = cv2.CALIB_USE_INTRINSIC_GUESS
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [all_3d_pos], [all_pixel_coords],
        img_shape, intrinsic_guess, None, flags=flags)
    print("calibrated camera intrinsic:\n", mtx)

    r = Rotation.from_rotvec(rvecs[0].reshape(-1))
    ext_R = r.as_matrix()
    ext = np.column_stack((ext_R, tvecs[0]))

    full_ext = np.row_stack((ext, [0, 0, 0, 1]))
    print("calibrated camera extrinsic:\n", full_ext)

    projM = mtx @ full_ext[:3]
    print("calibrated projection matrix:\n", projM)

    cameraTworld = np.linalg.inv(full_ext)
    print("calibrated camera to world transformation:\n", cameraTworld)

    print("camera 3d position:\n", cameraTworld[:3, 3])
    R_cTw = cameraTworld[0:3]
    R_cTw = R_cTw[:, :3]
    r = Rotation.from_matrix(R_cTw)
    camera_orient = r.as_quat()
    print("camera orientation (quarternion):\n", camera_orient)

    if args.visual_distribution:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(all_3d_pos[:, 0], all_3d_pos[:, 1], all_3d_pos[:, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    if VISUAL_REPROJ:
        for exp_id in use_for_calibration:
            gif = imageio.get_reader(target_dir + "/" + exp_id + ".gif")
            t = 0
            for img in gif:
                img = img[:, :, :3]
                img = cv2.resize(
                    img, (img.shape[1] * SCALE, img.shape[0] * SCALE))
                states = np.load(target_dir + "/states_" + exp_id + ".npy")
                state = states[t, :3]
                state = np.concatenate([state, [1]])
                print("state:", state)
                pix_3d = projM @ state
                pix_2d = np.array([pix_3d[0] / pix_3d[2], pix_3d[1] / pix_3d[2]])
                print(pix_2d)
                annotated = display_annotation(img, pix_2d)
                cv2.imwrite(target_dir + "/reproj_" + exp_id + "_" + str(t) + ".png", annotated)
                t += 1
                if t > 2:
                    break

if __name__ == "__main__":
    calibrate_directory()
    # calibrate_viewpoint()