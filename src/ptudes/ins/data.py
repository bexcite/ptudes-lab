from typing import Optional
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
        imu.avel = imu_packet.angular_vel
        if _intr_rot is not None:
            imu.lacc = _intr_rot @ imu.lacc
            imu.avel = _intr_rot @ imu.avel
        imu.dt = dt
        return imu


@dataclass
class NavState:
    pos: np.ndarray = np.zeros(3)    # Vec3
    # att_v: np.ndarray = np.zeros(3)
    att_q: np.ndarray = np.array([0, 0, 0, 1])  # Quat, xyzw
    # att_h: np.ndarray = np.eye(3)    # Mat3x3, SO(3)
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
             f"  att_v: {log_rot_mat(self.att_h)}\n"
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


def ekf_traj_ate(ekf_gt, ekf):
    """Calculate ATE for trajectories in update knots."""

    # assert len(ekf_gt._nav_scan_idxs) == len(ekf._nav_scan_idxs)

    # collect corresponding navs using update/scans knots
    t = []
    navs = []
    navs_gt = []

    nav_gt_it = iter(ekf_gt._navs[::-1])
    t_gt_it = iter(ekf_gt._navs_t[::-1])
    nav_gt = next(nav_gt_it)
    nav_gt_t = next(t_gt_it)
    for nav_idx in ekf._nav_scan_idxs[::-1]:
        n = ekf._navs[nav_idx]
        n_t = ekf._navs_t[nav_idx]
        t.append(n_t)
        navs.append(n)
        while nav_gt_t != n_t:
            nav_gt = next(nav_gt_it)
            nav_gt_t = next(t_gt_it)
        navs_gt.append(nav_gt)

    # print(f"{len(t) = }")
    # print(f"{len(navs) = }")
    # print(f"{len(navs_gt) = }")

    trans_d = []
    rot_d = []
    for nav, nav_gt in zip(navs, navs_gt):
        p1 = nav.pose_mat()
        p2 = nav_gt.pose_mat()
        trans_d.append(np.linalg.norm(p2[:3, 3] - p1[:3, 3]))
        rd = Rotation.from_matrix(
            np.transpose(p1[:3, :3]) @ p2[:3, :3]).as_rotvec()
        rot_d.append(np.linalg.norm(rd))
    ate_t = np.sum(np.square(trans_d)) / len(trans_d)
    ate_r = np.sum(np.square(rot_d)) / len(rot_d)
    ate_r = ate_r * 180 / np.pi


    return ate_r, ate_t
