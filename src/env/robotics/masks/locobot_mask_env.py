import os
from scipy.spatial.transform.rotation import Rotation
import numpy as np
import time
import imageio
import h5py
from pupil_apriltags import Detector
import cv2

from src.env.robotics.masks.base_mask_env import MaskEnv
from src.env.robotics.rotations import euler2quat, mat2quat, quat2euler
from src.env.robotics.robot_env import RobotEnv
from src.env.robotics.masks.locobot_analytical_ik import AnalyticInverseKinematics as AIK


class LocobotMaskEnv(MaskEnv):
    def __init__(self):
        model_path = os.path.join("locobot", "locobot.xml")
        initial_qpos = None
        n_actions = 1
        n_substeps = 1
        seed = None
        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
        self._img_width = 640
        self._img_height = 480
        self._camera_name = "main_cam"
        self._joints = [f"joint_{i}" for i in range(1, 6)]
        self._joints.append("gripper_revolute_joint")

    def compare_traj(self, traj_name, qpos_data, real_imgs):
        joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]
        # run qpos trajectory
        gif = []
        for i, qpos in enumerate(qpos_data):
            self.sim.data.qpos[joint_references] = qpos
            self.sim.forward()
            # self.render("human")
            img = self.render("rgb_array")
            mask = self.get_robot_mask()
            real_img = real_imgs[i]
            mask_img = real_img.copy()
            mask_img[mask] = (0, 255, 255)
            # mask_img = mask_img.astype(int)
            # mask_img[mask] += (100, 0, 0)
            # mask_img = mask_img.astype(np.uint8)
            comparison = mask_img
            # comparison = np.concatenate([img, real_img, mask_img], axis=1)
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{traj_name}_mask_" + str(i) + ".png", mask_img)
            gif.append(comparison)
        imageio.mimwrite(f"{traj_name}_mask.gif", gif)

    def get_robot_mask(self, width=None, height=None):
        """
        Return binary img mask where 1 = robot and 0 = world pixel.
        robot_mask_with_obj means the robot mask is computed with object occlusions.
        """
        # returns a binary mask where robot pixels are True
        seg = self.render("rgb_array", segmentation=True)  # flip the camera
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == self.mj_const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        if width is None or height is None:
            mask_dim = [self._img_height, self._img_width]
        else:
            mask_dim = [height, width]
        mask = np.zeros(mask_dim, dtype=np.bool)
        # TODO: change these to include the robot base
        ignore_parts = {"finger_r_geom", "finger_l_geom"}
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if name is not None:
                if name in ignore_parts:
                    continue
                mask[ids == i] = True
        return mask


def get_camera_pose_from_apriltag(image):
    detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray,
                              estimate_tag_pose=True,
                              camera_params=[612.45,
                                             612.45,
                                             330.55,
                                             248.61],
                              tag_size=0.0353)
    print("[INFO] {} total AprilTags detected".format(len(results)))

    # loop over the AprilTag detection results
    for r in results:
        pose_t = r.pose_t
        pose_R = r.pose_R
        print("pose_t", r.pose_t)
        print("pose_R", r.pose_R)
    return pose_t, pose_R


def predict_next_qpos(eef_curr, qpos_curr, action):
    """
    eef_curr: (3, ) 3d position of eef
    qpos_curr: (5, )
    action: (2, ) planar action
    """
    # TODO: record pitch/roll in eef pose in the future
    PUSH_HEIGHT = 0.15
    DEFAULT_PITCH = 1.3
    DEFAULT_ROLL = 0.0
    eef_next = np.zeros(3)
    eef_next[0:2] = eef_curr[0:2] + action
    eef_next[2] = PUSH_HEIGHT

    ik_solver = AIK()

    qpos_next = np.zeros(5)
    qpos_next[0:4] = ik_solver.ik(eef_next, alpha=-DEFAULT_PITCH, cur_arm_config=qpos_curr[0:4])
    qpos_next[4] = DEFAULT_ROLL
    return qpos_next


def overlay_trajs(traj_path1, traj_path2):
    with h5py.File(traj_path1 + ".hdf5", "r") as f:
        imgs1 = np.array(f['observations'])
    with h5py.File(traj_path2 + ".hdf5", "r") as f:
        imgs2 = np.array(f['observations'])
    avg_img = np.zeros(imgs1[0].shape)
    for t in range(imgs1.shape[0]):
        avg_img += imgs1[t]
    for t in range(imgs2.shape[0]):
        avg_img += imgs2[t]
    avg_img /= (imgs1.shape[0] + imgs2.shape[0])

    avg_img = avg_img.astype(np.uint8)

    avg_img = cv2.cvtColor(avg_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{traj_path1}_overlay.png", avg_img)


if __name__ == "__main__":
    """
    Load data:
    """

    data_path = "/mnt/ssd1/pallab/locobot_data/data_2021-03-07_03_48_46"

    traj_path1 = "/mnt/ssd1/pallab/locobot_data/data_2021-03-12_05_00_28"
    traj_path2 = "/mnt/ssd1/pallab/locobot_data/data_2021-03-12_16_47_37"

    # overlay_trajs(traj_path1, traj_path2)

    with h5py.File(data_path + ".hdf5", "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())

        qposes = np.array(f['qpos'])
        imgs = np.array(f['observations'])
        eef_states = np.array(f['states'])
        actions = np.array(f['actions'])

    K = 0
    predicted_Kstep_qpos = []
    for t in range(actions.shape[0] - K + 1):
        action_Kstep = np.sum(actions[t:t + K, 0:2], axis=0)
        qpos_next = predict_next_qpos(eef_states[t], qposes[t], action_Kstep)
        print("prediction:", qpos_next)
        print("real:", qposes[t + K])
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
    print("ar tag position:\n", env.sim.data.get_geom_xpos("ar_tag_geom"))
    print("ar tag orientation:\n", env.sim.data.get_geom_xmat("ar_tag_geom"))
    tagTbase = np.column_stack((env.sim.data.get_geom_xmat("ar_tag_geom"), env.sim.data.get_geom_xpos("ar_tag_geom")))
    tagTbase = np.row_stack((tagTbase, [0, 0, 0, 1]))
    print("tagTbase:\n", tagTbase)

    # tag to camera transformation
    pose_t, pose_R = get_camera_pose_from_apriltag(imgs[t])
    tagTcam = np.column_stack((pose_R, pose_t))
    tagTcam = np.row_stack((tagTcam, [0, 0, 0, 1]))
    print("tagTcam:\n", tagTcam)

    # tag in camera to tag in robot transformation
    # For explanation, refer to Kun's hand drawing
    tagcTtagw = np.array([[0, 0, -1, 0],
                          [0, -1, 0, 0],
                          [-1, 0, 0, 0],
                          [0, 0, 0, 1]])

    camTbase = tagTbase @ tagcTtagw @ np.linalg.inv(tagTcam)
    print("camTbase:\n", camTbase)

    rot_matrix = camTbase[:3, :3]
    cam_pos = camTbase[:3, 3]
    rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
    cam_rot = Rotation.from_matrix(rot_matrix) * rel_rot

    cam_id = 0
    offset = [0, 0, 0]
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

    env.sim.forward()

    env.compare_traj(data_path, predicted_Kstep_qpos, imgs[K:])

    while True:
        env.render("human")
