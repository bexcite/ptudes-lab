from typing import Optional, Iterable, Union

import numpy as np
from copy import deepcopy

from ouster.client import (SensorInfo, LidarScan, ChanField, FieldTypes,
                           UDPProfileLidar)
import ouster.client._client as _client
import ouster.client as client

from ouster.sdk.pose_util import exp_rot_vec

from dataclasses import dataclass

GRAV = 9.782940329221166

@dataclass
class NavState:
    pos: np.ndarray = np.zeros(3)    # Vec3
    att_h: np.ndarray = np.eye(3)    # Mat3x3, SO(3)
    vel: np.ndarray = np.zeros(3)    # Vec3

    bias_acc: np.ndarray = np.zeros(3) # Vec3
    bias_gyr: np.ndarray = np.zeros(3)  # Vec3

    # gravity vector?


@dataclass
class NavErrState:
    dpos: np.ndarray = np.zeros(3)    # Vec3
    datt_v: np.ndarray = np.zeros(3)  # Vec3, tangent-space
    dvel: np.ndarray = np.zeros(3)    # Vec3

    dbias_acc: np.ndarray = np.zeros(3) # Vec3
    dbias_gyr: np.ndarray = np.zeros(3) # Vec3


@dataclass
class IMU:
    lacc: np.ndarray = np.zeros(3)
    avel: np.ndarray = np.zeros(3)
    ts: float = 0
    dt: float = 0

    @staticmethod
    def from_packet(imu_packet: client.ImuPacket, prev_ts: Optional[float] = None) -> "IMU":
        imu = IMU()
        imu.ts = imu_packet.sys_ts / 10**9
        imu.lacc = GRAV * imu_packet.accel
        imu.avel = imu_packet.angular_vel
        dt = 0.01
        if prev_ts is not None:
            if imu.ts - prev_ts < dt:
                dt = imu.ts - prev_ts
        imu.dt = dt
        return imu


class LioEkf:

    def __init__(self, metadata: SensorInfo):
        self._metadata = metadata

        self._sensor_to_imu = np.linalg.inv(
            self._metadata.imu_to_sensor_transform)

        self._g_fn = GRAV * np.array([0, 0, -1])

        self._start_ts = -1

        # self._last_imup_ts = -1
        self._last_lidarp_ts = -1
        self._last_scan_ts = -1

        self._nav_curr = NavState()
        self._nav_prev = NavState()

        self._imu_prev = IMU()
        self._imu_curr = IMU()

        self._initialized = False

        print(f"init: nav_pre  = {self._nav_prev}")
        print(f"init: nav_curr = {self._nav_curr}")

    def _insMech(self):
        print(f"nav_prev = {self._nav_prev}")
        print(f"nav_curr = {self._nav_curr}")

        print(f"imu_prev = {self._imu_prev}")
        print(f"imu_curr = {self._imu_curr}")


        imucurr_vel = self._imu_curr.lacc * self._imu_curr.dt
        imucurr_angle = self._imu_curr.avel * self._imu_curr.dt
        imupre_vel = self._imu_prev.lacc * self._imu_prev.dt
        imupre_angle = self._imu_prev.avel * self._imu_prev.dt

        p1 = np.cross(imucurr_angle, imucurr_vel) / 2   #  1/2 * (w(k) x a(k) * s(k)^2)
        p2 = np.cross(imupre_angle, imucurr_vel) / 12   #  1/12 * (w(k-1) x a(k)) * s(k)^2
        p3 = np.cross(imupre_vel, imucurr_angle) / 12   #  1/12 * (a(k-1) x w(k)) * s(k)^2

        delta_v_fb = imucurr_vel + p1 + p2 + p3

        delta_v_fn = self._nav_prev.att_h @ delta_v_fb

        # gravity vel part
        delta_vgrav = self._g_fn * self._imu_curr.dt

        print(f"delta_grav = {delta_vgrav}")
        print(f"delta_v_fn  = {delta_v_fn}")

        self._nav_curr.vel = self._nav_prev.vel + delta_vgrav + delta_v_fn

        self._nav_curr.pos = self._nav_prev.pos + 0.5 * (self._nav_curr.vel + self._nav_prev.vel)

        rot_vec_b = imucurr_angle + np.cross(imupre_angle, imucurr_angle)
        self._nav_curr.att_h = self._nav_prev.att_h @ exp_rot_vec(rot_vec_b)

        print(f"after velocity: {self._nav_curr.vel = }")
        print(f"after position: {self._nav_curr.pos = }")
        print(f"after rotation:\n {self._nav_curr.att_h}\n")




    def processImuPacket(self, imu_packet: client.ImuPacket) -> None:
        imu_ts = imu_packet.sys_ts
        local_ts = self._check_start_ts(imu_ts) / 10**9

        self._imu_prev = self._imu_curr

        imu = IMU.from_packet(imu_packet, prev_ts=self._imu_prev.ts)
        self._imu_curr = imu

        # compensate imu/gyr
        self._imu_curr.lacc -= self._nav_curr.bias_acc
        self._imu_curr.avel -= self._nav_curr.bias_gyr

        if not self._initialized:
            self._initialized = True
            return

        self._insMech()

        self._nav_prev = deepcopy(self._nav_curr)


        # cov predict
        #

        # print(f"ts: {imu_ts}, local_ts: {local_ts:.6f}, "
        #       f"processImu: {self._imu_curr = }")
        
        print(f"processImu: acc={self._imu_curr.lacc}, gyr={self._imu_curr.avel}")
        
        print(f"      prev: acc={self._imu_prev.lacc}, gyr={self._imu_prev.avel}")

        # if self._last_imup_ts < 0:
        #     self._last_imup_ts = local_ts
        #     print("first IMU")
        #     return

        # self._last_imup_ts = local_ts

    def processLidarPacket(self, lidar_packet: client.LidarPacket) -> None:
        col_ts = lidar_packet.timestamp[0]
        local_ts = self._check_start_ts(col_ts) / 10**9
        # print(f"ts: {col_ts}, local_ts: {local_ts:.6f}, "
        #       f"processLidar: {lidar_packet = }")

    def processLidarScan(self, ls: client.LidarScan) -> None:
        ls_ts = client.first_valid_column_ts(ls)
        local_ts = self._check_start_ts(ls_ts) / 10**9
        # print(f"ts: {ls_ts}, local_ts: {local_ts:.6f}, "
        #       f"processScan: {ls = }")
        # print(f"{ls.timestamp = }")

    def _check_start_ts(self, ts: int) -> int:
        if self._start_ts < 0:
            self._start_ts = ts
        if ts < self._start_ts:
            print("OH!!!! ts = ", ts, ", start_ts = ", self._start_ts)
            return 0
        return ts - self._start_ts



class LioEkfScans(client.ScanSource):
    """LIO EKF experimental implementation"""

    def __init__(self, source: client.PacketSource,
                 *,
                 fields: Optional[FieldTypes] = None) -> None:
        self._source = source

        self._fields: Union[FieldTypes, UDPProfileLidar]
        self._fields = (fields if fields is not None else
                        self._source.metadata.format.udp_profile_lidar)

        self._lio_ekf = LioEkf(source.metadata)


    def __iter__(self) -> Iterable[LidarScan]:
        """Consume packets with odometry"""

        w = self._source.metadata.format.columns_per_frame
        h = self._source.metadata.format.pixels_per_column

        columns_per_packet = self._source.metadata.format.columns_per_packet

        ls_write = None
        batch = _client.ScanBatcher(self._source.metadata)

        it = iter(self._source)

        imu_cnt = 0

        while True:
            try:
                packet = next(it)
            except StopIteration:
                yield ls_write
                return

            if isinstance(packet, client.LidarPacket):
                self._lio_ekf.processLidarPacket(packet)

                ls_write = ls_write or LidarScan(h, w, self._fields, columns_per_packet)

                if batch(packet, ls_write):
                    # new scan finished
                    self._lio_ekf.processLidarScan(ls_write)
                    yield ls_write
                    ls_write = None

            elif isinstance(packet, client.ImuPacket):
                self._lio_ekf.processImuPacket(packet)
                imu_cnt += 1

            if imu_cnt > 20:
                exit(0)


    def close(self) -> None:
        self._source.close()

    @property
    def metadata(self) -> SensorInfo:
        return self._source.metadata
