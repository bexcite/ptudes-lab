from typing import Optional, Iterable, Union

import numpy as np

from ouster.client import (SensorInfo, LidarScan, ChanField, FieldTypes,
                           UDPProfileLidar)
import ouster.client._client as _client
import ouster.client as client

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
        imu.lacc = imu_packet.accel
        imu.avel = imu_packet.angular_vel
        dt = 0.1
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
        self._start_ts = -1

        # self._last_imup_ts = -1
        self._last_lidarp_ts = -1
        self._last_scan_ts = -1

        self._nav_curr = NavState()
        self._nav_prev = NavState()

        self._imu_prev = IMU()
        self._imu_curr = IMU()

        print(f"init: nav_pre  = {self._nav_prev}")
        print(f"init: nav_curr = {self._nav_curr}")

    def _insMech(self):
        pass

    def processImuPacket(self, imu_packet: client.ImuPacket) -> None:
        imu_ts = imu_packet.sys_ts
        local_ts = self._check_start_ts(imu_ts) / 10**9

        self._imu_prev = self._imu_curr

        imu = IMU.from_packet(imu_packet, prev_ts=self._imu_prev.ts)
        self._imu_curr = imu

        # compensate imu/gyr
        self._imu_curr.lacc -= self._nav_curr.bias_acc
        self._imu_curr.avel -= self._nav_curr.bias_gyr

        self._insMech()

        print(f"ts: {imu_ts}, local_ts: {local_ts:.6f}, "
              f"processImu: {self._imu_curr = }")
        print(f"  prev: {self._imu_prev = }")

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

            if imu_cnt > 5:
                exit(0)


    def close(self) -> None:
        self._source.close()

    @property
    def metadata(self) -> SensorInfo:
        return self._source.metadata
