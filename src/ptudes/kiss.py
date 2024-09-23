from typing import List, Optional

import numpy as np
from ouster.sdk import client
from ouster.sdk.client import (SensorInfo, LidarScan, ChanField)

from kiss_icp.config import load_config
# from kiss_icp.registration import get_registration
from kiss_icp.kiss_icp import KissICP
from kiss_icp.config.parser import KISSConfig

from ouster.sdk.util.pose_util import PoseH, log_pose
from scipy.spatial.transform import Rotation

Vec3 = np.ndarray


class KissICPWrapper:
    """Thin wrapper to use with Ouster SDK LidarScans objects"""

    def __init__(self,
                 metadata: SensorInfo,
                 *,
                 _min_range: float = 5,
                 _max_range: float = 100,
                 _use_extrinsics: bool = False):
        self._metadata = metadata
        self._xyz_lut = client.XYZLut(self._metadata,
                                      use_extrinsics=_use_extrinsics)

        w = self._metadata.format.columns_per_frame
        h = self._metadata.format.pixels_per_column

        self._timestamps = np.tile(np.linspace(0, 1.0, w, endpoint=False),
                                   (h, 1))

        self._max_range = _max_range
        self._min_range = _min_range

        self._kiss_config = load_config(None,
                                        deskew=True,
                                        max_range=self._max_range)
        self._kiss_config.data.min_range = self._min_range

        self._kiss = KissICP(config=self._kiss_config)

        # using last valid column timestamp as a pose ts
        self._poses_ts = []
        self._poses = []

        self._err_dt = []
        self._err_drot = []
        self._sigmas = []

    def register_frame(self,
                       scan: LidarScan,
                       initial_guess: Optional[PoseH] = None) -> PoseH:
        """Register scan with kiss icp"""

        sel_flag = scan.field(ChanField.RANGE) != 0
        xyz = self._xyz_lut(scan)[sel_flag]
        timestamps = self._timestamps[sel_flag]

        # TODO[pb]: This could be done differently ...
        ts = client.last_valid_column_ts(scan) * 1e-09

        self._kiss_register_frame(xyz,
                                  timestamps,
                                  ts,
                                  initial_guess=initial_guess)

        self._poses_ts.append(ts)

        return self.pose

    def deskew(self, frame, timestamps) -> np.ndarray:
        return self._kiss.compensator.deskew_scan(frame, self._kiss.poses,
                                                  timestamps)

    # Embed the KissICP original register_frame() in order to substitute the
    # prediction model/initial guess which is not exposed as an interface.
    # Could be a subclassing, but seems this way is also fine .... idk ...
    def _kiss_register_frame(self,
                             frame,
                             timestamps,
                             ts: float,
                             initial_guess: Optional[PoseH] = None):
        # Apply motion compensation
        kself = self._kiss
        frame = kself.compensator.deskew_scan(frame, timestamps,
                                              kself.last_delta)

        # Preprocess the input cloud
        frame = kself.preprocess(frame)

        # Voxelize
        source, frame_downsample = kself.voxelize(frame)

        # Get adaptive_threshold
        sigma = kself.adaptive_threshold.get_threshold()

        # Compute initial_guess for ICP
        if initial_guess is None:
            initial_guess = kself.last_pose @ kself.last_delta

        # Run ICP
        new_pose = self._kiss.registration.align_points_to_map(
            points=source,
            voxel_map=kself.local_map,
            initial_guess=initial_guess,
            max_correspondance_distance=3 * sigma,
            kernel=sigma / 3,
        )
        pose_gain = np.linalg.inv(initial_guess) @ new_pose

        dt = np.linalg.norm(pose_gain[:3, 3])
        drot = np.linalg.norm(
            Rotation.from_matrix(pose_gain[:3, :3]).as_rotvec())

        self._err_dt.append(dt)
        self._err_drot.append(drot)
        self._sigmas.append(sigma)

        # print(f"dt = {dt:.05f}, drot = {drot:.05f}")

        # Compute the difference between the prediction and the actual estimate
        model_deviation = np.linalg.inv(initial_guess) @ new_pose

        # Update step: threshold, local map, delta, and the last pose
        kself.adaptive_threshold.update_model_deviation(model_deviation)
        kself.local_map.update(frame_downsample, new_pose)
        kself.last_delta = np.linalg.inv(kself.last_pose) @ new_pose
        kself.last_pose = new_pose

        self.poses.append(new_pose)

        return frame, source

    @property
    def velocity(self) -> Vec3:
        """Get linear velocity estimate from kiss icp poses"""
        if len(self.poses) < 2:
            return np.zeros(3)
        prediction = self._kiss.last_delta
        dt = self.poses_ts[-1] - self.poses_ts[-2]
        return prediction[:3, 3] / dt

    @property
    def pose(self) -> PoseH:
        """Get the last pose"""
        if not self.poses:
            return np.eye(4)
        return self.poses[-1]

    @property
    def poses(self) -> List[PoseH]:
        """Get all poses"""
        return self._poses

    @property
    def poses_ts(self) -> List[float]:
        """Get all poses"""
        return self._poses_ts

    @property
    def local_map_points(self) -> np.ndarray:
        return self._kiss.local_map.point_cloud()

    @property
    def _config(self) -> KISSConfig:
        """Get underlying kiss icp config"""
        return self._kiss.config
