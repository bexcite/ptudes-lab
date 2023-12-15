from typing import Optional

from ouster.viz import PointViz, Cloud

import ouster.sdk.pose_util as pu

import numpy as np

INIT_POINT_CLOUD_SIZE = 300000

class PointCloud:
    """Helper to draw unstructured point cloud with PointViz"""

    def __init__(self,
                 point_viz: PointViz,
                 *,
                 pose: pu.PoseH = np.eye(4),
                 enabled: bool = True,
                 point_size: int = 1,
                 _init_size: int = INIT_POINT_CLOUD_SIZE):
        self._viz = point_viz
        self._pose = pose

        self._points = np.zeros((_init_size, 3), dtype=np.float32, order='F')
        self._keys = np.zeros(_init_size, dtype=np.float32)

        self._active_key = 0.7
        
        # next idx for the new points to add
        self._points_idx = 0

        self._cloud = Cloud(_init_size)
        self._cloud.set_point_size(point_size)

        self._enabled = False
        if enabled:
            self.enable()

    @property
    def enabled(self) -> bool:
        """True if cloud is added to the viz"""
        return self._enabled

    def enable(self) -> None:
        """Enable the cloud and add it to the viz if needed"""
        if not self._enabled:
            self._viz.add(self._cloud)
            self._enabled = True

    def disable(self) -> None:
        """Disable the cloud and remove it from the viz"""
        if self._enabled:
            self._viz.remove(self._cloud)
            self._enabled = False

    def toggle(self) -> bool:
        """Toggle the cloud visibility (i.e. presence in the viz)"""
        if not self._enabled:
            self.enable()
        else:
            self.disable()
        return self._enabled

    @property
    def pose(self) -> np.ndarray:
        """Cloud pose, 4x4 matrix"""
        return self._pose

    @pose.setter
    def pose(self, pose: np.ndarray):
        """Set cloud pose, 4x4 matrix, and update internal states"""
        self._pose = pose
        self.update()

    @property
    def points(self) -> str:
        """Cloud points"""
        return self._points[:self._points_idx]

    @points.setter
    def points(self, points: np.ndarray):
        """Set points, and update internal states"""
        n = points.shape[0]
        if n > self._points.shape[0]:
            print("NOT IMPLEMENTED Cloud grow")
            exit(0)
        self._points[:n] = points
        self._keys[:n] = self._active_key
        self._points[n:] = np.zeros([1, 3])
        self._keys[n:] = 0
        self._points_idx = n
        self.update()

    def update(self) -> None:
        """Update label component viz states."""
        self._cloud.set_pose(self._pose)
        self._cloud.set_xyz(self._points)
        self._cloud.set_key(self._keys[np.newaxis, ...])
