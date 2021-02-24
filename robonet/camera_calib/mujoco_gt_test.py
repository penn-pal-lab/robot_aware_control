import numpy as np
import h5py
import cv2
from scipy.spatial.transform import Rotation


def load_hdf5(path):
    demo = {}
    with h5py.File(path, "r") as hf:
        demo["actions"] = hf["actions"][:]
        demo["states"] = hf["states"][:]

        # x, y, z, quaternion
        # x: pointing inwards
        # y: pointing to the left
        # z:
        demo["robot_state"] = hf["robot_state"][:]
        demo["robot_demo"] = hf["robot_demo"][:]
        demo["0_camera_intrinsic"] = hf["0_camera_intrinsic"][:]
        demo["0_camera_pose"] = hf["0_camera_pose"][:]

        # (u, v), u is the horizontal dimension (from left to right) and v is the vertical dimension (from up to down)
        demo["0_eef_keypoints"] = hf["0_eef_keypoints"][:]
        for k, v in hf.items():
            if "object" in k:
                demo[k] = v[:]
    return demo


if __name__ == "__main__":
    pixel_coords = []
    eef_3d = []
    num_files_consider = 100
    img_shape = (64, 64)

    for file_id in range(num_files_consider):
        gt = load_hdf5("camera_calib/mujoco_gts/straight_push_0_" + str(file_id) + ".hdf5")
        if file_id == 0:
            # print("keys in this file:\n", gt.keys())
            print("images shape:", gt['object_inpaint_demo'].shape)
            print("gt camera intrinsic:\n", gt['0_camera_intrinsic'])
            
            print("gt camera pose:\n", gt['0_camera_pose'])
            r = Rotation.from_quat([gt['0_camera_pose'][4], gt['0_camera_pose'][5],
                                    gt['0_camera_pose'][6], gt['0_camera_pose'][3]])
            gt_cTw = np.column_stack((r.as_matrix(), gt['0_camera_pose'][:3]))
            # print("gt camera to world transformation:\n", gt_cTw)
            full_gt_cTw = np.row_stack((gt_cTw, [0, 0, 0, 1]))
            
            gt_ext = np.linalg.inv(full_gt_cTw)
            print("gt camera extrinsic:\n", gt_ext)

            gt_proj = gt['0_camera_intrinsic'] @ gt_ext[:3]
            print("gt projection matrix:\n", gt_proj)

        for t in range(gt['object_inpaint_demo'].shape[0]):
            # cv2.imwrite("camera_calib/mujoco_gts/scene_" + str(file_id) +
            #             "_" + str(t) + ".png", gt['object_inpaint_demo'][t])
            eef_img = gt['0_eef_keypoints'][t]
            # eef_img[0] = 64 - eef_img[0]
            pixel_coords.append(eef_img)
            # print(gt['robot_state'][t][:3])
            eef_3d.append(gt['robot_state'][t][:3])

    pixel_coords = np.array(pixel_coords, dtype=np.float32)
    eef_3d = np.array(eef_3d)

    # Use eef to calibrate camera intrisic and extrinsic
    print("------------calibration---------------")
    intrinsic_guess = np.array([[80.6, 0, 31.5],
                                [0, 80.6, 31.5],
                                [0, 0, 1]])
    flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [eef_3d], [pixel_coords], img_shape, intrinsic_guess, None, flags=flags)
    print("calibrated camera intrinsic:\n", mtx)

    r = Rotation.from_rotvec(rvecs[0].reshape(-1))
    ext_R = r.as_matrix()
    ext = np.column_stack((ext_R, tvecs[0]))

    full_ext = np.row_stack((ext, [0, 0, 0, 1]))
    print("calibrated camera extrinsic:\n", full_ext)

    projM = mtx @ full_ext[:3]
    print("calibrated projection matrix:\n", projM)

    cameraTworld = np.linalg.inv(full_ext)
    # print("calibrated camera to world transformation:\n", cameraTworld)
