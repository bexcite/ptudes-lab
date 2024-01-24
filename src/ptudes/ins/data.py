from typing import Optional, Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation

from dataclasses import dataclass

import ouster.client as client
from ouster.sdk.pose_util import log_rot_mat

GRAV = 9.782940329221166

@dataclass
class IMU:
    lacc: np.ndarray = np.zeros(3)
    avel: np.ndarray = np.zeros(3)
    ts: float = 0
    dt: float = 0

    @staticmethod
    def from_packet(imu_packet: client.ImuPacket,
                    dt: float = 0.01,
                    _intr_rot: Optional[np.ndarray] = None) -> "IMU":
        imu = IMU()
        imu.ts = imu_packet.sys_ts / 10**9
        imu.lacc = GRAV * imu_packet.accel
        imu.avel = np.pi * imu_packet.angular_vel / 180.0
        if _intr_rot is not None:
            imu.lacc = _intr_rot @ imu.lacc
            imu.avel = _intr_rot @ imu.avel
        imu.dt = dt
        return imu


@dataclass
class NavState:
    pos: np.ndarray = np.zeros(3)    # Vec3
    att_q: np.ndarray = np.array([0, 0, 0, 1])  # Quat, xyzw
    vel: np.ndarray = np.zeros(3)    # Vec3

    bias_gyr: np.ndarray = np.zeros(3)  # Vec3
    bias_acc: np.ndarray = np.zeros(3) # Vec3

    grav: np.ndarray = GRAV * np.array([0, 0, -1])

    update: bool = False

    # NavStateLog for viz/debug parts
    cov: Optional[np.ndarray] = None
    scan: Optional[client.LidarScan] = None

    xyz: Optional[np.ndarray] = None
    frame: Optional[np.ndarray] = None
    frame_ds: Optional[np.ndarray] = None
    source: Optional[np.ndarray] = None

    src: Optional[np.ndarray] = None
    src_hl: Optional[np.ndarray] = None
    src_source: Optional[np.ndarray] = None
    src_source_hl: Optional[np.ndarray] = None
    tgt: Optional[np.ndarray] = None
    tgt_hl: Optional[np.ndarray] = None

    kiss_pose: Optional[np.ndarray] = None
    kiss_map: Optional[np.ndarray] = None

    local_map: Optional[np.ndarray] = None

    # gravity vector?

    def pose_mat(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.att_h
        pose[:3, 3] = self.pos
        return pose

    @property
    def att_h(self):
        return Rotation.from_quat(self.att_q).as_matrix()

    @att_h.setter
    def att_h(self, val: np.ndarray):
        self.att_q = Rotation.from_matrix(val).as_quat()

    @property
    def att_v(self):
        return Rotation.from_quat(self.att_q).as_rotvec()

    @att_v.setter
    def att_v(self, val: np.ndarray):
        self.att_q = Rotation.from_rotvec(val).as_quat()

    def _formatted_str(self) -> str:
        sb = " (S)" if self.scan else ""
        s = (f"NavState{sb}:\n"
             f"  pos: {self.pos}\n"
             f"  vel: {self.vel}\n"
             f"  att_v: {self.att_v}\n"
             f"  bg: {self.bias_gyr}\n"
             f"  ba: {self.bias_acc}\n"
             f"  grav: {self.grav}\n")
        return s

    def __repr__(self) -> str:
        return self._formatted_str()


def set_blk(m: np.ndarray, row_id: int, col_id: int,
            b: np.ndarray) -> np.ndarray:
    br, bc = b.shape
    m[row_id:row_id + br, col_id:col_id + bc] = b
    return m


def blk(m: np.ndarray,
        row_id: int,
        col_id: int,
        nrows: int,
        ncols: Optional[int] = None) -> np.ndarray:
    if ncols is None:
        ncols = nrows
    return m[row_id:row_id + nrows, col_id:col_id + ncols]


def calc_ate(navs_poses, gt_poses) -> Tuple[float, float]:
    """Calculate Avg Traj Error (ATE)

    Time aligned, but not trasform. i.e. first poses of two trajectories
    will be aligned to the the same value before calculating ATE.
    
    Args:
      navs_poses: list of poses from the filter
      gt_poses: aligned ground truth poses

    Return:
      tuple with ATE_rotation (deg) and ATE_translation (m)
    """
    assert len(navs_poses) == len(gt_poses)
    assert len(navs_poses)

    pose0_inv = navs_poses[0] @ np.linalg.inv(gt_poses[0])

    trans_d = []
    rot_d = []
    for nav_pose, gt_pose in zip(navs_poses, gt_poses):
        gt_pose = pose0_inv @ gt_pose
        trans_d.append(np.linalg.norm(gt_pose[:3, 3] - nav_pose[:3, 3]))
        rd = Rotation.from_matrix(
            np.transpose(nav_pose[:3, :3]) @ gt_pose[:3, :3]).as_rotvec()
        rot_d.append(np.linalg.norm(rd))
    ate_t = np.sum(np.square(trans_d)) / len(trans_d)
    ate_r = np.sum(np.square(rot_d)) / len(rot_d)
    ate_r = ate_r * 180 / np.pi
    return ate_r, ate_t


def calc_ate_from_navs(navs, gt_poses) -> Tuple[float, float]:
    """Calculate Avg Traj Error (ATE)
    
    Args:
      navs: list of NavStates from the filter
      gt_poses: aligned ground truth poses

    Return:
      tuple with ATE_rotation (deg) and ATE_translation (m)
    """
    nav_poses = [nav.pose_mat() for nav in navs]
    return calc_ate(nav_poses, gt_poses)


def _collect_navs_from_gt(ekf_gt, ekf) -> Tuple[List, List, List]:
    # collect corresponding navs using update/scans knots
    t = []
    navs = []
    navs_gt = []

    nav_gt_it = iter(ekf_gt._navs[::-1])
    t_gt_it = iter(ekf_gt._navs_t[::-1])
    nav_gt = next(nav_gt_it)
    nav_gt_t = next(t_gt_it)
    for nav_idx in ekf._nav_update_idxs[::-1]:
        n = ekf._navs[nav_idx]
        n_t = ekf._navs_t[nav_idx]
        t.append(n_t)
        navs.append(n)
        while nav_gt_t != n_t:
            nav_gt = next(nav_gt_it)
            nav_gt_t = next(t_gt_it)
        navs_gt.append(nav_gt)

    t = t[::-1]
    navs = navs[::-1]
    navs_gt = navs_gt[::-1]
    return (t, navs_gt, navs)


def ekf_traj_ate(ekf_gt, ekf):
    """Calculate ATE for trajectories in update knots."""

    t, navs_gt, navs = _collect_navs_from_gt(ekf_gt, ekf)

    nav_poses = [nav.pose_mat() for nav in navs]
    gt_poses = [nav.pose_mat() for nav in navs_gt]

    return calc_ate(nav_poses, gt_poses)


class StreamStatsTracker:
    """Tracks the mean/std stats for scans range"""

    def __init__(self,
                 use_beams_num: Optional[int] = None,
                 metadata: Optional[client.SensorInfo] = None):
        # Ouster metadata
        self._metadata = metadata
        self._mean = 0
        self._scans_num = 0
        self._points_num = 0
        self._sigma_sq = 0
        self._use_beams_num = use_beams_num
        self._beams_sel = None

        self._mean_acc = np.zeros(3)
        self._mean_gyr = np.zeros(3)

        # sigma^2 * n - accumulator
        self._sigman_acc = np.zeros(3)
        self._sigman_gyr = np.zeros(3)
        self._imu_num = 0

        self._max_ts = 0
        self._min_ts = 0

        self._min_range = 0
        self._max_range = 0

        self._all = np.array([])

    def __range_to_m(self, range: np.ndarray) -> np.ndarray:
        """Convert Ouster range data to meters

        NOTE: better to use Ouster XYZLut, but it's slower if we
        don't care about the all points in 3D
        """
        range_to_m_coefs = 0.001
        if self._metadata:
            if (self._metadata.format.udp_profile_lidar ==
                    client.UDPProfileLidar.PROFILE_LIDAR_RNG15_RFL8_NIR8):
                range_to_m_coefs = 8 * range_to_m_coefs
        return range * range_to_m_coefs

    def _track_min_max_ts(self, ts: float):
        if not self._imu_num and not self._scans_num:
            self._min_ts = ts
            self._max_ts = ts
        else:
            self._min_ts = min(self._min_ts, ts)
            self._max_ts = max(self._max_ts, ts)

    def _track_min_max_range(self, range: np.ndarray):
        if not self._points_num:
            self._min_range = np.min(range)
            self._max_range = np.max(range)
        else:
            self._min_range = min(self._min_range, np.min(range))
            self._max_range = max(self._max_range, np.max(range))

    def trackImu(self, imu: IMU):
        """Update IMU mean/sigma"""
        mean_acc_prev = self._mean_acc.copy()
        mean_gyr_prev = self._mean_gyr.copy()

        self._mean_acc += (imu.lacc - self._mean_acc) / (self._imu_num + 1)
        self._sigman_acc += (imu.lacc - mean_acc_prev) * (imu.lacc -
                                                          self._mean_acc)

        self._mean_gyr += (imu.avel - self._mean_gyr) / (self._imu_num + 1)
        self._sigman_gyr += (imu.avel - mean_gyr_prev) * (imu.avel -
                                                          self._mean_gyr)

        self._track_min_max_ts(imu.ts)

        self._imu_num += 1

    def trackScan(self, ls: client.LidarScan) -> Tuple[float, float]:
        """Update scan stats for range mean/sigma"""
        if self._use_beams_num:
            if self._beams_sel is None:
                self._beams_sel = np.linspace(0,
                                              ls.h,
                                              num=self._use_beams_num,
                                              endpoint=False,
                                              dtype=int)
            range = ls.field(client.ChanField.RANGE)[self._beams_sel, :]
        else:
            range = ls.field(client.ChanField.RANGE)

        range = range[range > 0]

        # range to meters
        range = self.__range_to_m(range)

        self._track_min_max_range(range)

        m = np.mean(range)
        n = range.size
        v = np.var(range)

        # some reference: https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
        s1 = 0 if not self._points_num else (self._points_num - 1) * self._sigma_sq
        corr = self._points_num * n * np.square((self._mean - m)) / (
            (self._points_num + n) * ((self._points_num + n - 1)))
        self._sigma_sq = (s1 + n * v) / (self._points_num + n - 1) + corr

        self._mean = (self._mean * self._points_num + m * n) / (self._points_num + n)

        # NOTE: tracking here the sensor timestamp and not the
        #       udp/host timestamp, can be parameterised later
        #       when needed
        self._track_min_max_ts(client.last_valid_column_ts(ls) * 1e-9)

        self._scans_num += 1
        self._points_num += n

    @property
    def range_mean(self) -> float:
        return self._mean

    @property
    def range_std(self) -> float:
        return np.sqrt(self._sigma_sq)

    @property
    def acc_mean(self) -> np.ndarray:
        return self._mean_acc

    @property
    def acc_std(self) -> np.ndarray:
        return np.sqrt(self._sigman_acc / self._imu_num)

    @property
    def gyr_mean(self) -> np.ndarray:
        return self._mean_gyr

    @property
    def gyr_std(self) -> np.ndarray:
        return np.sqrt(self._sigman_gyr / self._imu_num)

    @property
    def dt(self) -> float:
        return self._max_ts - self._min_ts

    def _formatted_str(self) -> str:
        s3_min_range = max(self._min_range,
                           self.range_mean - 3 * self.range_std)
        s3_max_range = min(self._max_range,
                           self.range_mean + 3 * self.range_std)
        s = (
            f"StreamStatsTracker[dt: {self.dt:.04f} s, imus: {self._imu_num}, scans: {self._scans_num}]:\n"
            f"  range_mean: {self.range_mean:.03f} m,\n"
            f"  range_std: {self.range_std:.03f} m (s3 span: [{s3_min_range:.03f} - {s3_max_range:.03f} m])\n"
            f"  range min max: {self._min_range:.03f} - {self._max_range:.03f} m\n"
            f"  acc_mean: {self.acc_mean} m/s^2\n"
            f"  acc_std: {self.acc_std}\n"
            f"  gyr_mean: {self.gyr_mean} rad/s\n"
            f"  gyr_std: {self.gyr_std}")
        return s

    def __repr__(self):
        return self._formatted_str()