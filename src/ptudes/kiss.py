from typing import List

import numpy as np
import ouster.client as client
from ouster.client import (SensorInfo, LidarScan, ChanField)

from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP
from kiss_icp.config.parser import KISSConfig

from ouster.sdk.pose_util import PoseH

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
        self._kiss = KissICP(config=self._kiss_config)

        # using last valid column timestamp as a pose ts
        self._poses_ts = []

    def register_frame(self, scan: LidarScan) -> PoseH:
        """Register scan with kiss icp"""

        sel_flag = scan.field(ChanField.RANGE) != 0
        xyz = self._xyz_lut(scan)[sel_flag]
        timestamps = self._timestamps[sel_flag]

        self._kiss.register_frame(xyz, timestamps)

        # TODO[pb]: This could be done differently ...
        ts = client.last_valid_column_ts(scan) * 1e-09
        self._poses_ts.append(ts)

        return self.pose

    def deskew(self, frame, timestamps) -> np.ndarray:
        return self._kiss.compensator.deskew_scan(frame, self._kiss.poses,
                                                  timestamps)

    @property
    def velocity(self) -> Vec3:
        """Get linear velocity estimate from kiss icp poses"""
        if len(self.poses) < 2:
            return np.zeros(3)
        prediction = self._kiss.get_prediction_model()
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
        return self._kiss.poses

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
