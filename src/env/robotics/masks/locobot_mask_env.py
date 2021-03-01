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


class LocobotMaskEnv(MaskEnv):
    def __init__(self):
        model_path = os.path.join("locobot", "locobot.xml")
        initial_qpos = None
        n_actions = 1
        n_substeps = 1
        seed = None
        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
        self._img_width = 85
        self._img_height = 64
        self._camera_name = "main_cam"
        self._joints = [f"joint_{i}" for i in range(1, 6)]
        self._joints.append("gripper_revolute_joint")

    def compare_traj(self, traj_name, qpos_data, gripper_data, real_imgs):
        joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]
        # run qpos trajectory
        gif = []
        for i, qpos in enumerate(qpos_data):
            self.sim.data.qpos[joint_references] = qpos
            grip_state = gripper_data[i]
            eef_pos = grip_state[:3]
            eef_site = self.sim.model.body_name2id("eef_body")
            self.sim.model.body_pos[eef_site] = eef_pos
            self.sim.forward()
            # self.render("human")
            img = self.render("rgb_array")
            mask = self.get_robot_mask()
            real_img = real_imgs[i]
            mask_img = real_img.copy()
            mask_img[mask] = (0, 255, 255)
            comparison = mask_img
            # comparison = np.concatenate([img, real_img, mask_img], axis=1)
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
        ignore_parts = {"base_link_vis", "base_link_col", "head_vis"}
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if name is not None:
                if name in ignore_parts:
                    continue
                a = "vis" in name
                b = "col" in name
                c = "gripper" in name
                d = "mesh" in name
                if any([a, b, c, d]):
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
                              tag_size=0.035)
    print("[INFO] {} total AprilTags detected".format(len(results)))

    # loop over the AprilTag detection results
    for r in results:
        pose_t = r.pose_t
        pose_R = r.pose_R
        print("pose_t", r.pose_t)
        print("pose_R", r.pose_R)
    return pose_t, pose_R


if __name__ == "__main__":
    """
    Load data:
    """
    data_path = "/mnt/ssd1/pallab/locobot_data/data_2-24-18-03.hdf5"

    with h5py.File(data_path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())

        qposes = np.array(f['qpos'])
        imgs = np.array(f['observations'])

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
    env.sim.data.qpos[env._joint_references] = qposes[t]
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

    while True:
        env.render("human")
