import numpy as np
from typing import Optional, Iterable, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod

import ouster.client as client
from ouster.viz import (PointViz, Label, ScansAccumulator)

import ouster.sdk.pose_util as pu
from ptudes.utils import estimate_apex_dolly

PoseH = np.ndarray

def update_min_max(cmm: np.array, new_point: np.array):
    cmm[:, 0] = np.minimum(cmm[:, 0], new_point)
    cmm[:, 1] = np.maximum(cmm[:, 1], new_point)


class FlyingState(Enum):
    """Camera movement states"""
    BUILDING = 0
    TO_THE_BEGINNING = 1
    COURSING = 2
    TO_THE_APEX = 3


class FState(ABC):
    """Flyby states handler"""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def update(self, dt: float, viz: PointViz) -> Optional[FlyingState]:
        """Progress the state and return the next on transition"""
        pass


class BuildingState(FState):

    def __init__(self,
                 scan_source: Iterable[client.LidarScan],
                 scans_accum: ScansAccumulator,
                 start_scan: Optional[int] = None,
                 end_scan: Optional[int] = None,
                 min_max: Optional[np.ndarray] = None,
                 poses: Optional[List[Tuple[float, PoseH]]] = None,
                 next_state: Optional[FlyingState] = None) -> None:
        super().__init__("BUILDING")

        if start_scan and end_scan:
            assert start_scan <= end_scan, "start_scan should be lte end_scan"
        self._start_scan = start_scan
        self._end_scan = end_scan

        self._source = scan_source

        self._scans_accum = scans_accum

        if min_max is None:
            self._min_max = np.zeros((3, 2))
        else:
            assert min_max.shape == (3, 2)
            self._min_max = min_max

        # return the processed scans ts
        self._poses = poses
        assert len(self._poses) == 0, "poses list should be empty on init"

        self._next_state = next_state

        self._scans_iter = iter(self._source)
        self._scan_idx = 0

    def update(self, dt: float, viz: PointViz) -> Optional[FlyingState]:
        # paused state
        if dt == 0.0:
            return None

        try:
            if self._start_scan:
                self._lookup_start_scan(viz)

            scan = self._next_scan()

            # draw map points
            self._scans_accum.update(scan)
            self._scans_accum.draw(update=False)

            if self._poses is not None:
                col_idx = client.last_valid_column(scan)
                self._poses.append(
                    (scan.timestamp[col_idx], scan.pose[col_idx]))

            # move camera along with dolly
            scan_pose = client.first_valid_column_pose(scan).copy()
            scan_pose[:3, :3] = np.eye(3)
            camera_target = np.linalg.inv(scan_pose)
            viz.camera.set_target(camera_target)

            update_min_max(self._min_max, scan_pose[:3, 3])
            dolly = estimate_apex_dolly(self._min_max, viz.camera.get_fov())

            curr_dolly = viz.camera.get_dolly()
            new_dolly = curr_dolly + 0.05 * (dolly - curr_dolly)
            viz.camera.set_dolly(new_dolly)

        except StopIteration:
            return self._next_state

        return None

    def _lookup_start_scan(self, pviz: PointViz) -> None:
        temp_osd = None
        try:
            while self._start_scan and self._scan_idx < self._start_scan:
                self._next_scan()
                if temp_osd is None:
                    temp_osd = Label("", 1, 1, align_right=True)
                    pviz.add(temp_osd)
                # (HACK) show text status in temp Label
                # we can do this only during BUILDING state while
                # there is no main right bottom OSD label is set.
                osd_str = f"looking up start scan {self._start_scan} : {self._scan_idx}"
                temp_osd.set_text(osd_str)
                pviz.update()
        finally:
            if temp_osd is not None:
                pviz.remove(temp_osd)

    def _next_scan(self) -> Optional[client.LidarScan]:
        """Fetches the next scan to process or StopIteraton on end"""
        if self._scan_idx > self._end_scan:
            raise StopIteration
        scan = next(self._scans_iter)
        self._scan_idx += 1
        return scan


class CameraTransitionState(FState):
    """Transition camera to the target and/or pitch/dolly/yaw"""
    def __init__(self,
                 name: str = "",
                 target: Optional[PoseH] = None,
                 pitch: Optional[float] = -70,
                 dolly: Optional[float] = -70,
                 yaw: Optional[float] = 120,
                 velocity: float = 1.0,
                 next_state: Optional[FlyingState] = None) -> None:
        super().__init__(name)
        self._target = target
        self._pitch = pitch
        self._dolly = dolly
        self._yaw = yaw
        self._next_state = next_state
        self._t = 0
        self._v = velocity

    def update(self, dt: float, viz: PointViz) -> Optional[FlyingState]:
        curr_pitch = viz.camera.get_pitch()
        curr_dolly = viz.camera.get_dolly()
        curr_yaw = viz.camera.get_yaw()

        dt = self._v * dt

        if self._t + dt >= 1.0:
            return self._next_state

        if self._target is not None:
            curr_target = viz.camera.get_target()
            curr_target = np.array(curr_target).reshape((4,4)).transpose()
            pose_d = pu.log_pose(curr_target @ self._target)
            new_target = np.linalg.inv(curr_target) @ pu.exp_pose6(
                dt / (1 - self._t) * pose_d)
            viz.camera.set_target(np.linalg.inv(new_target))

        if self._pitch is not None:
            pitch_new = curr_pitch + dt / (1 - self._t) * (self._pitch -
                                                           curr_pitch)
            viz.camera.set_pitch(pitch_new)

        if self._dolly is not None:
            dolly_new = curr_dolly + dt / (1 - self._t) * (self._dolly -
                                                           curr_dolly)
            viz.camera.set_dolly(dolly_new)

        if self._yaw is not None:
            yaw_new = curr_yaw + dt / (1 - self._t) * (self._yaw - curr_yaw)
            viz.camera.set_yaw(yaw_new)

        self._t += dt

        return None


class CoursingState(FState):
    """Follow the scans poses (i.e. set target along the traj eval poses)"""
    def __init__(self,
                 traj_eval: pu.TrajectoryEvaluator,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None,
                 velocity: float = 1.0,
                 min_duration: float = 3.0,
                 next_state: Optional[FlyingState] = None) -> None:
        super().__init__("COURSING")
        assert len(traj_eval._poses) > 0, "traj_eval should have some poses"
        self._start_t = traj_eval._poses[0][
            0] if start_time is None else start_time
        self._end_t = traj_eval._poses[-1][0] if end_time is None else end_time
        self._traj_eval = traj_eval

        self._v = velocity
        # slowing down velocity so we have at least min_duration coursing time
        if (self._v > 0 and self._end_t - self._start_t > 0
                and (self._end_t - self._start_t) / self._v < min_duration):
            self._v = (self._end_t - self._start_t) / min_duration

        self._next_state = next_state

        self._t = self._start_t

    def update(self, dt: float, viz: PointViz) -> Optional[FlyingState]:
        dt = self._v * dt

        if self._t + dt > self._end_t:
            return self._next_state

        new_target = self._traj_eval.pose_at(self._t)
        viz.camera.set_target(np.linalg.inv(new_target))

        self._t += dt

        return None
