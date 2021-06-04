import os
import numpy as np
import time
import h5py
import imageio
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation
from pupil_apriltags import Detector
from tqdm import tqdm
from src.env.robotics.masks.locobot_analytical_ik import AnalyticInverseKinematics as AIK


from src.env.robotics.masks.locobot_mask_env import (LocobotMaskEnv,
                                                     get_camera_pose_from_apriltag, load_states,
                                                     predict_next_qpos,
                                                     load_data)

total_files = 1000


if __name__ == "__main__":
    """
    Load data:
    """
    ik_solver = AIK()
    # data_path = "/mnt/ssd1/pallab/locobot_data/data_2021-03-12/"
    # data_path = "/scratch/anonymous/Robonet/locobot_views/c0"
    data_path = "/home/pallab/locobot_ws/src/eef_control/data/locobot_modified_views/c0/"

    detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    n_files = 0
    skipped_files = []
    """
    Init Mujoco env:
    """
    env = LocobotMaskEnv()

    env._joints = [f"joint_{i}" for i in range(1, 6)]
    env._joint_references = [
        env.sim.model.get_joint_qpos_addr(x) for x in env._joints
    ]
    for filename in tqdm(os.listdir(data_path)):
        if filename.endswith(".hdf5"):
            overwrite = False
            # with h5py.File(os.path.join(data_path, filename), "r") as f:
            #     if 'masks' in f.keys():
            #         # TODO: overwrite for now, change to continue in the future
            #         overwrite = True

            # print(os.path.join(data_path, filename))

            qposes, imgs, eef_states, actions = load_data(os.path.join(data_path, filename))
            # eef_states = load_states(os.path.join(data_path, filename))
            # if qposes is None or imgs is None or eef_states is None or actions is None:
            #     skipped_files.append(filename)
            #     continue

            K = 0
            predicted_Kstep_qpos = []
            for t in range(actions.shape[0] - K + 1):
                action_Kstep = np.sum(actions[t:t + K, 0:2], axis=0)
                qpos_next = predict_next_qpos(eef_states[t], qposes[t], action_Kstep)
                predicted_Kstep_qpos.append(qpos_next)
            predicted_Kstep_qpos = np.stack(predicted_Kstep_qpos)

            # generate qpos from eef pos using AIK
            qposes = []
            qpos_curr = [0] * 4
            for pos in eef_states:
                DEFAULT_PITCH = 1.3
                DEFAULT_ROLL = 0.0
                eef_next=  pos


                qpos_next = np.zeros(5)
                qpos_next[0:4] = ik_solver.ik(eef_next, alpha=-DEFAULT_PITCH, cur_arm_config=qpos_curr)
                qpos_next[4] = DEFAULT_ROLL
                qpos_curr = qpos_next
                qposes.append(qpos_next)
            qposes = np.asarray(qposes)

            # Use mujoco to get eef states
            for i, qpos in enumerate(qposes):
                pos = env.get_gripper_pos(qpos)
                eef_states[i] = pos

            # >>>>>>>>>>> K step rollout for analytical masks!

            # K = 31
            # predicted_Kstep_qpos = [qposes[0]]
            # qpos_next = qposes[0]
            # eef_next = eef_states[0]
            # for t in range(K - 1):
            #     qpos_next = predict_next_qpos(eef_next, qpos_next, actions[t, :2])
            #     eef_next = env.get_gripper_pos(qpos_next)
            #     predicted_Kstep_qpos.append(qpos_next)
            # predicted_Kstep_qpos = np.stack(predicted_Kstep_qpos)

            """
            camera params:
            """
            # if no Apriltag detected, use next image
            # tag_detected = False
            # for t in range(qposes.shape[0]):
            #     # tag to camera transformation
            #     pose_t, pose_R = get_camera_pose_from_apriltag(imgs[t], detector=detector)
            #     if pose_t is None or pose_R is None:
            #         continue

            #     tag_detected = True

            #     target_qpos = qposes[t]
            #     env.sim.data.qpos[env._joint_references] = target_qpos
            #     env.sim.forward()

            #     # tag to base transformation
            #     tagTbase = np.column_stack((env.sim.data.get_geom_xmat("ar_tag_geom"),
            #                                 env.sim.data.get_geom_xpos("ar_tag_geom")))
            #     tagTbase = np.row_stack((tagTbase, [0, 0, 0, 1]))

            #     tagTcam = np.column_stack((pose_R, pose_t))
            #     tagTcam = np.row_stack((tagTcam, [0, 0, 0, 1]))

            #     # tag in camera to tag in robot transformation
            #     # For explanation, refer to anonymous's hand drawing
            #     tagcTtagw = np.array([[0, 0, -1, 0],
            #                           [0, -1, 0, 0],
            #                           [-1, 0, 0, 0],
            #                           [0, 0, 0, 1]])

            #     camTbase = tagTbase @ tagcTtagw @ np.linalg.inv(tagTcam)

            #     rot_matrix = camTbase[:3, :3]
            #     cam_pos = camTbase[:3, 3]
            #     rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
            #     cam_rot = Rotation.from_matrix(rot_matrix) * rel_rot

            #     cam_id = 0
            #     offset = [0, -0.007, 0.02]
            #     env.sim.model.cam_pos[cam_id] = cam_pos + offset
            #     cam_quat = cam_rot.as_quat()
            #     env.sim.model.cam_quat[cam_id] = [
            #         cam_quat[3],
            #         cam_quat[0],
            #         cam_quat[1],
            #         cam_quat[2],
            #     ]
            #     print("camera pose:")
            #     print(env.sim.model.cam_pos[cam_id])
            #     print(env.sim.model.cam_quat[cam_id])

            #     env.sim.forward()
            #     break

            # if not tag_detected:
            #     continue
            camTbase = np.array([[ 0.0452768,   0.73303716 ,-0.67868 ,    0.79116035],
                                [ 0.99869241 ,-0.01707084 , 0.04818772, -0.00249282],
                                [ 0.02373775 ,-0.67997435 ,-0.73285156 , 0.64026054],
                                [ 0.      ,    0.  ,        0.    ,      1.        ]])

            offset = [0, -0.015, 0.0125]
            env.set_opencv_camera_pose("main_cam", camTbase, offset)


            """
            compute masks
            """
            masks = []
            for i, qpos in enumerate(predicted_Kstep_qpos):
                env.sim.data.qpos[env._joint_references] = qpos
                env.sim.forward()
                mask = env.get_robot_mask()
                masks.append(mask)
                # imgs[i][mask] = (0, 255, 0)

            masks = np.stack(masks)
            # imageio.mimwrite(f"{filename}_masks.gif", imgs)
            # env.compare_traj(filename, qposes, eef_states, imgs)

            with h5py.File(os.path.join(data_path, filename), "a") as f:
                if overwrite:
                    f['masks'][:] = masks
                else:
                    f.create_dataset('masks', data=masks)
                # replace recorded qpos with AIK derived qpos
                f["qpos"][...] = qposes

            n_files += 1
            if n_files > total_files:
                break
    print("skipped", skipped_files)