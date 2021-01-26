import numpy as np
from src.env.robotics.robot_env import RobotEnv
from scipy.spatial.transform.rotation import Rotation


class MaskEnv(RobotEnv):
    def set_opencv_camera_pose(self, cam_name, camera_extrinsics):
        cam_id = self.sim.model.camera_name2id(cam_name)
        rot_matrix = camera_extrinsics[:3, :3]
        cam_pos = camera_extrinsics[:3, 3]
        rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
        cam_rot = Rotation.from_matrix(rot_matrix) * rel_rot
        offset = [0, 0, 0]
        self.sim.model.cam_pos[cam_id] = cam_pos + offset
        cam_quat = cam_rot.as_quat()
        self.sim.model.cam_quat[cam_id] = [
            cam_quat[3],
            cam_quat[0],
            cam_quat[1],
            cam_quat[2],
        ]
        self.sim.forward()

    def render(self, mode, segmentation=False):
        if mode == "rgb_array":
            out = super().render(
                mode,
                width=self._img_width,
                height=self._img_height,
                camera_name=self._camera_name,
                segmentation=segmentation,
            )
            return out[::-1, ::-1]
        elif mode == "human":
            super().render(mode)

    def get_robot_mask(self):
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
        mask_dim = [self._img_height, self._img_width]
        mask = np.zeros(mask_dim, dtype=np.bool)
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if name is not None:
                mask[ids == i] = self._is_robot_geom(name)
        return mask

    def _sample_goal(self):
        pass

    def _get_obs(self):
        return {"observation": np.array([0])}

    def _is_robot_geom(self, name):
        return NotImplementedError()

    def generate_masks(self,  qpos_data):
        joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]
        masks = []
        for qpos in qpos_data:
            self.sim.data.qpos[joint_references] = qpos
            self.sim.forward()
            mask = self.get_robot_mask()
            masks.append(mask)
        masks = np.asarray(masks, dtype=np.bool)
        return masks
