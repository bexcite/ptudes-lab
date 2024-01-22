from typing import Optional

from ouster.viz import PointViz, Cloud

import ouster.sdk.pose_util as pu

import numpy as np


RED_COLOR = np.array([1.0, 0.1, 0.1, 1.0])  # RGBA
BLUE_COLOR = np.array([0.4, 0.4, 1.0, 1.0])  # RGBA
YELLOW_COLOR = np.array([0.1, 1.0, 1.0, 1.0])  # RGBA
GREY_COLOR = np.array([0.5, 0.5, 0.5, 1.0])  # RGBA
GREY1_COLOR = np.array([0.7, 0.7, 0.7, 1.0])  # RGBA
WHITE_COLOR = np.array([1.0, 1.0, 1.0, 1.0])  # RGBA

INIT_POINT_CLOUD_SIZE = 10000


class PointCloud:
    """Helper to draw unstructured point cloud with PointViz"""

    def __init__(self,
                 point_viz: PointViz,
                 *,
                 pose: pu.PoseH = np.eye(4),
                 enabled: bool = True,
                 point_size: int = 1,
                 point_color: Optional[np.ndarray] = None,
                 _init_size: int = INIT_POINT_CLOUD_SIZE):
        self._viz = point_viz
        self._pose = pose
        self._point_size = point_size

        self._points = np.zeros((_init_size, 3), dtype=np.float32, order='F')
        self._keys = np.zeros(_init_size, dtype=np.float32)
        self._mask = np.zeros((_init_size, 4), dtype=float)
        if point_color is not None and point_color.size == 4:
            self._mask_color = point_color
        else:
            self._mask_color = np.zeros(4)

        self._active_key = 0.7

        # next idx for the new points to add
        self._points_idx = 0

        self._cloud = Cloud(_init_size)
        self._cloud.set_point_size(self._point_size)

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
            new_size = int(points.shape[0] * 1.3)
            new_points = np.zeros_like(self._points, shape=(new_size, 3))
            new_points[:self._points.shape[0]] = self._points
            self._points = new_points
            new_keys = np.zeros_like(self._keys, shape=(new_size))
            new_keys[:self._keys.shape[0]] = self._keys
            self._keys = new_keys
            new_mask = np.zeros_like(self._mask, shape=(new_size, 4))
            new_mask[:self._mask.shape[0]] = self._mask
            self._mask = new_mask
        self._points[:n] = points
        self._keys[:n] = self._active_key
        self._points[n:] = np.zeros([1, 3])
        self._keys[n:] = 0
        self._mask[:n] = self._mask_color
        self._mask[n:] = 0
        self._points_idx = n
        self.update()

    def update(self) -> None:
        """Update label component viz states."""
        if self._cloud.size < self._points.shape[0]:
            self._viz.remove(self._cloud)
            del self._cloud
            self._cloud = Cloud(self._points.shape[0])
            self._cloud.set_point_size(self._point_size)
            if self._enabled:
                self._viz.add(self._cloud)
        self._cloud.set_pose(self._pose)
        self._cloud.set_xyz(self._points)
        self._cloud.set_key(self._keys[np.newaxis, ...])
        self._cloud.set_mask(self._mask)


