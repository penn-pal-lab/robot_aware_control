import os
import numpy as np
import time
import imageio
import h5py
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation
from pupil_apriltags import Detector


from src.env.robotics.masks.locobot_mask_env import (LocobotMaskEnv,
                                                     get_camera_pose_from_apriltag,
                                                     predict_next_qpos,
                                                     load_data)

total_files = 100


if __name__ == "__main__":
    """
    Load data:
    """

    data_path = "/mnt/ssd1/pallab/locobot_data/data_2021-03-12/"

    n_files = 0
    pixel_changes = []
    camera_positions = []

    detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    for filename in os.listdir(data_path):
        if filename.endswith(".hdf5"):
            print(os.path.join(data_path, filename))

            qposes, imgs, eef_states, actions = load_data(os.path.join(data_path, filename))
            if qposes is None or imgs is None or eef_states is None or actions is None:
                continue

            K = 0
            predicted_Kstep_qpos = []
            for t in range(actions.shape[0] - K + 1):
                action_Kstep = np.sum(actions[t:t + K, 0:2], axis=0)
                qpos_next = predict_next_qpos(eef_states[t], qposes[t], action_Kstep)
                predicted_Kstep_qpos.append(qpos_next)
            predicted_Kstep_qpos = np.stack(predicted_Kstep_qpos)

            """
            Init Mujoco env:
            """
            env = LocobotMaskEnv()

            env._joints = [f"joint_{i}" for i in range(1, 6)]
            env._joint_references = [
                env.sim.model.get_joint_qpos_addr(x) for x in env._joints
            ]

            """
            camera params:
            """
            t = 1
            target_qpos = qposes[t]
            env.sim.data.qpos[env._joint_references] = target_qpos
            env.sim.forward()

            # tag to base transformation
            tagTbase = np.column_stack((env.sim.data.get_geom_xmat("ar_tag_geom"),
                                        env.sim.data.get_geom_xpos("ar_tag_geom")))
            tagTbase = np.row_stack((tagTbase, [0, 0, 0, 1]))

            # tag to camera transformation
            pose_t, pose_R = get_camera_pose_from_apriltag(imgs[t], detector=detector)
            if pose_t is None or pose_R is None:
                continue

            tagTcam = np.column_stack((pose_R, pose_t))
            tagTcam = np.row_stack((tagTcam, [0, 0, 0, 1]))

            # tag in camera to tag in robot transformation
            # For explanation, refer to Kun's hand drawing
            tagcTtagw = np.array([[0, 0, -1, 0],
                                  [0, -1, 0, 0],
                                  [-1, 0, 0, 0],
                                  [0, 0, 0, 1]])

            camTbase = tagTbase @ tagcTtagw @ np.linalg.inv(tagTcam)

            rot_matrix = camTbase[:3, :3]
            cam_pos = camTbase[:3, 3]
            rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
            cam_rot = Rotation.from_matrix(rot_matrix) * rel_rot

            cam_id = 0
            offset = [0, -0.01, 0.01]
            env.sim.model.cam_pos[cam_id] = cam_pos + offset
            cam_quat = cam_rot.as_quat()
            env.sim.model.cam_quat[cam_id] = [
                cam_quat[3],
                cam_quat[0],
                cam_quat[1],
                cam_quat[2],
            ]
            print("camera pose:")
            print(env.sim.model.cam_pos[cam_id])
            print(env.sim.model.cam_quat[cam_id])

            camera_positions.append(np.array(env.sim.model.cam_pos[cam_id]))

            env.sim.forward()

            # env.compare_traj(data_path, predicted_Kstep_qpos, imgs[K:])

            """
            Compute interaction rate
            TODO: apply mask, then compute # of changed pixels 
            """
            # for t in range(1, imgs.shape[0]):
            #     last_img = imgs[t - 1].astype(int)
            #     curr_img = imgs[t].astype(int)
            #     diff = np.abs(last_img - curr_img)
            #     pixel_diff = np.average(diff)
            #     # print(pixel_diff)
            #     pixel_changes.append(pixel_diff)

            n_files += 1
            if n_files > total_files:
                break

    """
    camera params std
    """
    camera_positions = np.stack(camera_positions)
    cam_pose_mean = np.mean(camera_positions, axis=0)
    cam_pose_var = np.std(camera_positions, axis=0)
    print("cam_pose_mean:", cam_pose_mean)
    print("cam_pose_var:", cam_pose_var)

    # plt.figure()
    # plt.hist(pixel_changes)
    # plt.xlabel("average pixel changes")
    # plt.ylabel("number of frames")
    # plt.show()
