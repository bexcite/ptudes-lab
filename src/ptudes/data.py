from typing import Optional
import numpy as np

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
    att_h: np.ndarray = np.eye(3)    # Mat3x3, SO(3)
    vel: np.ndarray = np.zeros(3)    # Vec3

    bias_gyr: np.ndarray = np.zeros(3)  # Vec3
    bias_acc: np.ndarray = np.zeros(3) # Vec3

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

    def _formatted_str(self) -> str:
        sb = " (S)" if self.scan else ""
        s = (f"NavState{sb}:\n"
             f"  pos: {self.pos}\n"
             f"  vel: {self.vel}\n"
             f"  att_v: {log_rot_mat(self.att_h)}\n"
             f"  bg: {self.bias_gyr}\n"
             f"  ba: {self.bias_acc}\n")
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
