from typing import Optional, Iterable, Union

import numpy as np
from copy import deepcopy

from ouster.client import (SensorInfo, LidarScan, ChanField, FieldTypes,
                           UDPProfileLidar)
import ouster.client._client as _client
import ouster.client as client

from ouster.sdk.pose_util import exp_rot_vec, log_rot_mat, _no_scipy_log_rot_mat

from kiss_icp.config import KISSConfig
from kiss_icp.config import load_config
from kiss_icp.pybind import kiss_icp_pybind
from kiss_icp.voxelization import voxel_down_sample
from kiss_icp.mapping import get_voxel_hash_map
from kiss_icp.threshold import get_threshold_estimator
from kiss_icp.registration import register_frame
from kiss_icp.kiss_icp import KissICP

import matplotlib.pyplot as plt

from dataclasses import dataclass

GRAV = 9.782940329221166
DEG2RAD = np.pi / 180.0

from .imu_test import imu_raw

def scan_begin_ts(ls: client.LidarScan) -> float:
    return client.first_valid_column_ts(ls) / 10**9

@dataclass
class NavState:
    pos: np.ndarray = np.zeros(3)    # Vec3
    att_h: np.ndarray = np.eye(3)    # Mat3x3, SO(3)
    vel: np.ndarray = np.zeros(3)    # Vec3

    bias_gyr: np.ndarray = np.zeros(3)  # Vec3
    bias_acc: np.ndarray = np.zeros(3) # Vec3

    # gravity vector?

    def pose_mat(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.att_h
        pose[:3, 3] = self.pos
        return pose


@dataclass
class NavErrState:
    dpos: np.ndarray = np.zeros(3)    # Vec3
    datt_v: np.ndarray = np.zeros(3)  # Vec3, tangent-space
    dvel: np.ndarray = np.zeros(3)    # Vec3

    dbias_gyr: np.ndarray = np.zeros(3) # Vec3
    dbias_acc: np.ndarray = np.zeros(3) # Vec3


@dataclass
class IMU:
    lacc: np.ndarray = np.zeros(3)
    avel: np.ndarray = np.zeros(3)
    ts: float = 0
    dt: float = 0

    @staticmethod
    def from_packet(imu_packet: client.ImuPacket, dt: float = 0.01, _intr_rot: Optional[np.ndarray] = None) -> "IMU":
        imu = IMU()
        imu.ts = imu_packet.sys_ts / 10**9
        imu.lacc = GRAV * imu_packet.accel
        imu.avel = imu_packet.angular_vel
        if _intr_rot is not None:
            imu.lacc = _intr_rot @ imu.lacc
            imu.avel = _intr_rot @ imu.avel
        imu.dt = dt
        return imu


def set_blk(m: np.ndarray, row_id: int, col_id: int,
            b: np.ndarray) -> np.ndarray:
    br, bc = b.shape
    m[row_id:row_id + br, col_id:col_id + bc] = b
    return m

def vee(vec: np.ndarray) -> np.ndarray:
    w = np.zeros((3, 3))
    w[0, 1] = -vec[2]
    w[0, 2] = vec[1]
    w[1, 0] = vec[2]
    w[1, 2] = -vec[0]
    w[2, 0] = -vec[1]
    w[2, 1] = vec[0]
    return w

class LioEkf:

    STATE_RANK = 15
    POS_ID = 0
    VEL_ID = 3
    PHI_ID = 6
    BG_ID = 9
    BA_ID = 12

    def __init__(self, metadata: SensorInfo):
        self._metadata = metadata

        self._sensor_to_imu = np.linalg.inv(
            self._metadata.imu_to_sensor_transform)

        self._metadata.extrinsic = self._sensor_to_imu
        self._xyz_lut = client.XYZLut(self._metadata, use_extrinsics=True)

        # self._imu_intr_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self._imu_intr_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        self._g_fn = GRAV * np.array([0, 0, -1])

        self._initpos_std = np.diag([0.05, 0.05, 0.05])
        self._initvel_std = np.diag([0.05, 0.05, 0.05])
        self._initatt_std = DEG2RAD * np.diag([1.0, 1.0, 0.5])

        # super good
        # self._acc_vrw = 50 * 1e-3 * GRAV  # m/s^2
        # self._gyr_arw = 1 * DEG2RAD  #  rad/s
        # self._acc_bias_std = 135 * 1e-6 * GRAV  # m/s^2 / sqrt(Hz)
        # self._gyr_bias_std = 0.005 * DEG2RAD  # rad/s / sqrt(Hz)

        self._acc_vrw = 50 * 1e-3 * GRAV  # m/s^2
        self._gyr_arw = 10 * DEG2RAD  #  rad/s
        self._acc_bias_std = 135 * 1e-6 * GRAV  # m/s^2 / sqrt(H)
        self._gyr_bias_std = 0.01 * DEG2RAD  # rad/s / sqrt(H)
        self._imu_corr_time = 3600  # s

        self._scan_meas_std = 0.02  # 2 cm (OS-0 on 100% refl target)

        # covariance for error-state Kalman filter
        self._cov = np.zeros((self.STATE_RANK, self.STATE_RANK))
        set_blk(self._cov, self.POS_ID, self.POS_ID,
                np.square(self._initpos_std))
        set_blk(self._cov, self.VEL_ID, self.VEL_ID,
                np.square(self._initvel_std))
        set_blk(self._cov, self.PHI_ID, self.PHI_ID,
                np.square(self._initatt_std))
        set_blk(self._cov, self.BG_ID, self.BG_ID,
                np.square(self._gyr_bias_std * np.eye(3)))
        set_blk(self._cov, self.BA_ID, self.BA_ID,
                np.square(self._acc_bias_std * np.eye(3)))

        self._cov_imu_bnoise = np.zeros((self.STATE_RANK, self.STATE_RANK))
        corr_coef = 2 / self._imu_corr_time
        set_blk(self._cov_imu_bnoise, self.BG_ID, self.BG_ID,
                corr_coef * np.square(self._gyr_bias_std * np.eye(3)))
        set_blk(self._cov_imu_bnoise, self.BA_ID, self.BA_ID,
                corr_coef * np.square(self._acc_bias_std * np.eye(3)))

        self._cov_mnoise = np.zeros((6, 6))
        set_blk(self._cov_mnoise, 0, 0, np.square(self._acc_vrw * np.eye(3)))
        set_blk(self._cov_mnoise, 3, 3, np.square(self._gyr_arw * np.eye(3)))

        # point measurement covariance (Er)
        self._cov_scan_meas = np.square(self._scan_meas_std * np.eye(3))
        self._cov_scan_meas_inv = np.linalg.inv(self._cov_scan_meas)

        # error-state (delta X)
        self._nav_err = NavErrState()
        self._reset_nav_err()

        # print(f"_cov = \n", self._cov.diagonal())

        self._imu_idx = 0
        self._scan_idx = 0

        self._start_ts = -1

        # self._last_imup_ts = -1
        self._last_lidarp_ts = -1
        self._last_scan_ts = -1

        self._nav_curr = NavState()

        # total_imu: accel =  [-0.93531433  0.36841277  0.23654302]
        # total_imu: avel  =  [ 1.26553045 -0.25072862 -2.18688865]
        # self._nav_curr.bias_acc = np.array([-0.93531433, 0.36841277, 0.23654302])
        # self._nav_curr.bias_gyr = np.array([ 1.26553045, -0.25072862, -2.18688865])

        # self._nav_curr.bias_acc = np.array([0.93531433, -0.36841277, 0.23654302])
        # self._nav_curr.bias_gyr = np.array([-1.26553045, 0.25072862, -2.18688865])

        # self._nav_curr.bias_acc = np.array([ -0.93531433,  -0.36841277, -19.80242368])
        # self._nav_curr.bias_gyr = np.array([1.26553045, 0.25072862, 2.18688865])

        # self._nav_prev = deepcopy(self._nav_curr)

        self._reset_nav()



        self._imu_prev = IMU()
        self._imu_curr = IMU()

        kiss_icp_config = load_config(None, deskew=True, max_range = 100)
        self._kiss_icp = KissICP(config=kiss_icp_config)
        self._local_map = get_voxel_hash_map(self._kiss_icp.config)
        self._adaptive_threshold = get_threshold_estimator(self._kiss_icp.config)
        print("kiss_icp = ", kiss_icp_config)
        print("adaptive_threshold = ", self._adaptive_threshold)
        # exit(0)

        self._navs = []  # nav states (full, after the update on scan)
        self._navs_t = []

        self._imu_total = IMU()
        self._imu_total_cnt = 0

        self._initialized = False

        self._lg_t = []
        self._lg_acc = []
        self._lg_gyr = []
        self._lg_pos = []
        self._lg_vel = []
        self._lg_scan = []
        self._lg_dsp = []

        print(f"init: nav_pre  = {self._nav_prev}")
        print(f"init: nav_curr = {self._nav_curr}")

    def _reset_nav(self):
        self._nav_curr.pos = np.zeros(3)
        self._nav_curr.vel = np.zeros(3)
        self._nav_curr.att_h = np.eye(3)
        self._nav_prev = deepcopy(self._nav_curr)
        print("------ RESET --- NAV -----")

    def _reset_nav_err(self):
        self._nav_err.dpos = np.zeros(3)
        self._nav_err.dvel = np.zeros(3)
        self._nav_err.datt_v = np.zeros(3)
        self._nav_err.dbias_gyr = np.zeros(3)
        self._nav_err.dbias_acc = np.zeros(3)
        print("------ RESET --- NAV -- ERR-----")

    def _insMech(self):
        # print(f"nav_prev = {self._nav_prev}")
        # print(f"nav_curr = {self._nav_curr}")

        # print(f"imu_prev = {self._imu_prev}")
        # print(f"imu_curr = {self._imu_curr}")

        sk = self._imu_curr.dt


        imucurr_vel = self._imu_curr.lacc * sk
        imucurr_angle = self._imu_curr.avel * sk
        imupre_vel = self._imu_prev.lacc * sk
        imupre_angle = self._imu_prev.avel * sk

        p1 = np.cross(imucurr_angle, imucurr_vel) / 2   #  1/2 * (w(k) x a(k) * s(k)^2)
        p2 = np.cross(imupre_angle, imucurr_vel) / 12   #  1/12 * (w(k-1) x a(k)) * s(k)^2
        p3 = np.cross(imucurr_angle, imupre_vel) / 12   #  1/12 * (a(k-1) x w(k)) * s(k)^2

        delta_v_fb = imucurr_vel + p1 + p2 - p3

        delta_v_fn = self._nav_prev.att_h @ delta_v_fb

        # gravity vel part
        delta_vgrav = self._g_fn * sk

        print(f"delta_grav = {delta_vgrav}")
        print(f"delta_v_fn  = {delta_v_fn}")

        self._nav_curr.vel = self._nav_prev.vel + delta_vgrav + delta_v_fn

        self._nav_curr.pos = self._nav_prev.pos + 0.5 * (self._nav_curr.vel +
                                                         self._nav_prev.vel)

        rot_vec_b = imucurr_angle + np.cross(imupre_angle, imucurr_angle) / 12
        self._nav_curr.att_h = self._nav_prev.att_h @ exp_rot_vec(rot_vec_b)

        # rvec = log_rot_mat(self._nav_curr.att_h)
        # rvec2 = _no_scipy_log_rot_mat(self._nav_curr.att_h)
        # print("rmat0 = \n", self._nav_curr.att_h)
        # print("rmat1 = \n", exp_rot_vec(rvec))
        # print("rvec0 = ", rvec)
        # print("rvec1 = ", log_rot_mat(exp_rot_vec(rvec)))
        # print("rvec2 = ", rvec2)
        # if (n := np.linalg.norm(rvec) > np.pi):
        #     rvec3 = (n - np.pi) / n * rvec
        #     print("rvec3 = ", rvec3)




        # print(f"after velocity: {self._nav_curr.vel = }")
        # print(f"after position: {self._nav_curr.pos = }")
        # print(f"after rotation:\n {self._nav_curr.att_h}\n")


    def processImuPacket(self, imu_packet: client.ImuPacket) -> None:
        # imu_ts = imu_packet.sys_ts
        # local_ts = self._check_start_ts(imu_ts) / 10**9

        # self._imu_prev = self._imu_curr

        imu = IMU.from_packet(imu_packet, _intr_rot=self._imu_intr_rot)
        # imu.dt = imu.ts - self._imu_prev.ts
        self.processImu(imu)


    def processImu(self, imu: IMU) -> None:

        self._imu_prev = self._imu_curr

        imu.dt = imu.ts - self._imu_prev.ts
        print(f"IMU[{self._imu_idx}] = ", imu)
        self._imu_idx += 1

        self._imu_curr = imu

        # compensate imu/gyr
        self._imu_curr.lacc -= self._nav_curr.bias_acc
        self._imu_curr.avel -= self._nav_curr.bias_gyr

        # accumulate total
        self._imu_total.lacc += self._imu_curr.lacc
        self._imu_total.avel += self._imu_curr.avel
        self._imu_total_cnt += 1

        if not self._initialized:
            self._initialized = True
            return

        self._insMech()

        sk = self._imu_curr.dt

        # update error-state (delta X)
        # TODO: Use real R(k-1)!
        dk = np.cross(self._nav_prev.att_h @ self._imu_curr.lacc,
                      self._nav_err.datt_v)
        self._nav_err.dpos = self._nav_err.dpos + self._nav_err.dvel * sk

        self._nav_err.dvel = (
            self._nav_err.dvel + sk * dk +
            sk * self._nav_prev.att_h @ self._nav_err.dbias_acc)

        self._nav_err.datt_v = (
            self._nav_err.datt_v - sk *
            (self._nav_prev.att_h @ self._nav_err.dbias_gyr))

        print("nav_err = \n", self._nav_err)

        Ak = np.eye(self.STATE_RANK)
        set_blk(Ak, self.POS_ID, self.VEL_ID, sk * np.eye(3))
        set_blk(Ak, self.VEL_ID, self.PHI_ID,
                sk * vee(self._nav_prev.att_h @ self._imu_curr.lacc))
        set_blk(Ak, self.VEL_ID, self.BA_ID, sk * self._nav_prev.att_h)
        set_blk(Ak, self.PHI_ID, self.BG_ID, -1.0 * sk * self._nav_prev.att_h)

        Bk = np.zeros((self.STATE_RANK, 6))
        set_blk(Bk, self.VEL_ID, 0,
                -sk * vee(self._nav_err.datt_v) @ self._nav_prev.att_h)

        self._cov = (Ak @ self._cov @ Ak.transpose() +
                     Bk @ self._cov_mnoise @ Bk.transpose() +
                     self._cov_imu_bnoise)

        np.set_printoptions(precision=3, linewidth=180)
        print("UPDATED COV:::::::::::::::::::::\n", self._cov)


        # logging
        self._lg_t += [self._imu_curr.ts]
        self._lg_acc += [self._imu_curr.lacc]
        self._lg_gyr += [self._imu_curr.avel]
        self._lg_pos += [self._nav_curr.pos]
        self._lg_vel += [self._nav_curr.vel]



        self._nav_prev = deepcopy(self._nav_curr)



        # test only when processLidarScan is disabled
        if not self._navs:
            self._reset_nav()
        self._navs += [deepcopy(self._nav_curr)]
        self._navs_t += [self._imu_curr.ts]


        # cov predict
        #

        # print(f"ts: {imu_ts}, local_ts: {local_ts:.6f}, "
        #       f"processImu: {self._imu_curr = }")

        # print(f"processImu: acc={self._imu_curr.lacc}, gyr={self._imu_curr.avel}")
        # print(f"      prev: acc={self._imu_prev.lacc}, gyr={self._imu_prev.avel}")

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
        ls_ts = scan_begin_ts(ls)

        # get XYZ points
        timestamps = np.tile(np.linspace(0, 1.0, ls.w, endpoint=False), (ls.h, 1))
        # filtering our zero returns makes it substantially faster for kiss-icp
        sel_flag = ls.field(ChanField.RANGE) != 0
        xyz = self._xyz_lut(ls)[sel_flag]
        timestamps = timestamps[sel_flag]
        print("ls.size = ", xyz.shape)
        print("timestamps.size = ", timestamps.shape)

        # deskew
        frame = self.deskew_scan(xyz, timestamps)

        source, frame_downsample = self.voxelize(frame)
        print("source.shape = ", source.shape)
        print("frame_downsample.shape = ", frame_downsample.shape)


        sigma = self.get_sigma_threshold()

        # update state

        # move source to map frams using imu integration of meas between
        # prev and current lidar scan
        Rk = self._nav_curr.att_h
        tk = self._nav_curr.pos
        exp_datt = exp_rot_vec(self._nav_err.datt_v)
        # if src.size:
        print("tk = ", tk)
        print("dtk = ", self._nav_err.dpos)
        print("exp_datt = ", exp_datt)


        h_dx = np.matmul(
            exp_rot_vec(self._nav_err.datt_v) @ Rk,
            source.transpose()).transpose() + tk + self._nav_err.dpos
        print("============================= hdx = ", h_dx.shape)
        print("h_dx_10 = ", h_dx[:10], np.linalg.norm(h_dx[0]))

        src, tgt = self._local_map.get_correspondences(source, 3 * sigma)
        print("src = ", src.shape)
        print("tgt = ", tgt.shape)
        if src.size:
            print("src_10 = ", src[:10], np.linalg.norm(src[0]))
            print("tgt_10 = ", tgt[:10], np.linalg.norm(tgt[0]))


        resid = src - tgt

        Ji = np.zeros((3, self.STATE_RANK))
        sum_Ji = np.zeros((self.STATE_RANK, self.STATE_RANK))
        sum_res = np.zeros(self.STATE_RANK)
        set_blk(Ji, 0, 0, np.eye(3))
        for i in range(src.shape[0]):
            set_blk(Ji, 0, self.PHI_ID, vee(1.0 * Rk @ src[i]))
            sum_Ji += Ji.transpose() @ self._cov_scan_meas_inv @ Ji
            sum_res += Ji.transpose() @ self._cov_scan_meas_inv @ resid[i]
            # print("ooooo i = ", i)
            # print("sum_res = \n", sum_res)


        cov_inv = np.linalg.inv(self._cov)
        H_plus_cov = np.linalg.inv(sum_Ji + cov_inv)
        delta_x = H_plus_cov @ sum_res

        print("delta_x = ", delta_x)

        # apply correction
        self._nav_err.dpos += delta_x[self.POS_ID:self.POS_ID + 3]
        self._nav_err.dvel += delta_x[self.VEL_ID:self.VEL_ID + 3]
        self._nav_err.datt_v += delta_x[self.PHI_ID:self.PHI_ID + 3]
        self._nav_err.dbias_gyr += delta_x[self.BG_ID:self.BG_ID + 3]
        self._nav_err.dbias_acc += delta_x[self.BA_ID:self.BA_ID + 3]

        self._cov = (np.eye(self.STATE_RANK) - H_plus_cov @ sum_Ji) @ self._cov

        print("_nav_err FINAL       = ", self._nav_err)

        initial_guess = self._nav_curr.pose_mat()

        new_icp_pose = register_frame(
            points=source,
            voxel_map=self._local_map,
            initial_guess=initial_guess,
            max_correspondance_distance=3 * sigma,
            kernel=sigma / 3,
        )
        print("_nav_curr FINAL (ICP) = ", new_icp_pose)

        self._nav_curr.pos += self._nav_err.dpos
        self._nav_curr.vel += self._nav_err.dvel
        self._nav_curr.att_h = exp_rot_vec(self._nav_err.datt_v) @ self._nav_curr.att_h
        self._nav_curr.bias_gyr += self._nav_err.dbias_gyr
        self._nav_curr.bias_acc += self._nav_err.dbias_acc

        print("_nav_curr FINAL = ", self._nav_curr.pose_mat())

        new_pose = self._nav_curr.pose_mat()

        self._adaptive_threshold.update_model_deviation(
            np.linalg.inv(initial_guess) @ new_pose)

        self._reset_nav_err()


        self._local_map.update(frame_downsample, new_pose)

        if not self._navs:
            self._reset_nav()

        self._navs += [deepcopy(self._nav_curr)]
        self._navs_t += [scan_begin_ts(ls)]

        print("navs = \n", self._navs[-3:])
        print("navs_t = \n", self._navs_t[-3:])

        print("local_map.size = ", self._local_map.point_cloud().shape)

        # input()

        # exit(0)

        self._nav_prev = deepcopy(self._nav_curr)



        self._lg_scan += [ls]
        self._lg_dsp += [(scan_begin_ts(ls), self._nav_curr.pos)]
        # self._reset_nav()


    def deskew_scan(self, xyz: np.ndarray,
                    timestamps: np.ndarray) -> np.ndarray:
        if len(self._navs) < 1:
            return xyz
        deskew_frame = kiss_icp_pybind._deskew_scan(
            frame=kiss_icp_pybind._Vector3dVector(xyz),
            timestamps=timestamps,
            start_pose=self._navs[-1].pose_mat(),
            finish_pose=self._nav_curr.pose_mat(),
        )
        return np.asarray(deskew_frame)

    def voxelize(self, orig_frame) -> np.ndarray:
        frame_downsample = voxel_down_sample(
            orig_frame, self._kiss_icp.config.mapping.voxel_size * 0.5)
        source = voxel_down_sample(frame_downsample,
                                   self._kiss_icp.config.mapping.voxel_size)
        return source, frame_downsample

    def get_sigma_threshold(self) -> float:
        adaptive = (self._kiss_icp.config.adaptive_threshold.initial_threshold
                    if not self.has_moved() else
                    self._adaptive_threshold.get_threshold())
        print("ADAOTIVE ==== ", adaptive)
        return adaptive

    def has_moved(self):
        if len(self._navs) < 1:
            return False
        compute_motion = lambda T1, T2: np.linalg.norm((np.linalg.inv(T1) @ T2)[:3, -1])
        motion = compute_motion(self._navs[0].pose_mat(), self._navs[-1].pose_mat())
        return motion > 5 * self._kiss_icp.config.adaptive_threshold.min_motion_th

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
                 fields: Optional[FieldTypes] = None,
                 _start_scan: Optional[int] = None,
                 _end_scan: Optional[int] = None) -> None:
        self._source = source

        self._start_scan = _start_scan or 0
        self._end_scan = _end_scan

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

        imu_idx = 0

        scan_idx = 0

        while True:
            try:
                packet = next(it)
            except StopIteration:
                yield ls_write
                break

            if isinstance(packet, client.LidarPacket):
                if scan_idx >= self._start_scan:
                    self._lio_ekf.processLidarPacket(packet)

                ls_write = ls_write or LidarScan(h, w, self._fields, columns_per_packet)

                if batch(packet, ls_write):
                    # new scan finished
                    if scan_idx >= self._start_scan:
                        self._lio_ekf.processLidarScan(ls_write)

                    yield ls_write

                    if (self._end_scan is not None
                            and scan_idx >= self._end_scan):
                        break

                    scan_idx += 1

                    ls_write = None

            elif isinstance(packet, client.ImuPacket):
                if self._start_scan == 0 or scan_idx >= self._start_scan - 1:

                    # acc_x = 0.1 if imu_idx < 10 else 0
                    # acc_y = 0.0 if imu_idx < 10 else 0.1

                    # acc = [acc_x, acc_y, GRAV]
                    # gyr = [0, 0, 5.0]
                    # imu = IMU(acc, gyr, imu_idx * 0.01)

                    # imu_r = imu_raw[imu_idx, :]
                    # acc = imu_r[:3] * GRAV
                    # gyr = imu_r[3:]
                    # imu = IMU(acc, gyr, imu_idx * 0.001)
                    
                    # self._lio_ekf.processImu(imu)

                    self._lio_ekf.processImuPacket(packet)

                imu_idx += 1

            # if imu_cnt > 150:
            #     break

        print(f"Finished: imu_idx = {imu_idx}, "
              f"scan_idx = {scan_idx}, "
              f"scans_num = {len(self._lio_ekf._lg_scan)}")

        print(
            "total_imu: accel = ",
            self._lio_ekf._imu_total.lacc / self._lio_ekf._imu_total_cnt +
            self._lio_ekf._g_fn)
        print("total_imu: avel  = ",
              self._lio_ekf._imu_total.avel / self._lio_ekf._imu_total_cnt)
        print(f"total_cnt = ", self._lio_ekf._imu_total_cnt)


        min_ts = self._lio_ekf._lg_t[0]
        print(f"imu_ts: {min_ts}")
        if self._lio_ekf._lg_scan:
            scan_first_ts = scan_begin_ts(self._lio_ekf._lg_scan[0])
            print(f"scan_ts: {scan_first_ts}")
            min_ts = min(min_ts, scan_first_ts)
        print(f"min_ts res: {min_ts}")


        # show graphs
        # print(f"lg_t = {self._lio_ekf._lg_t}")
        # print(f"lg_accel = {self._lio_ekf._lg_acc}")

        # plt.cla()


        t = [t - min_ts for t in self._lio_ekf._lg_t]
        acc_x = [a[0] for a in self._lio_ekf._lg_acc]
        acc_y = [a[1] for a in self._lio_ekf._lg_acc]
        acc_z = [a[2] for a in self._lio_ekf._lg_acc]

        gyr_x = [a[0] for a in self._lio_ekf._lg_gyr]
        gyr_y = [a[1] for a in self._lio_ekf._lg_gyr]
        gyr_z = [a[2] for a in self._lio_ekf._lg_gyr]

        pos_x = [a[0] for a in self._lio_ekf._lg_pos]
        pos_y = [a[1] for a in self._lio_ekf._lg_pos]
        pos_z = [a[2] for a in self._lio_ekf._lg_pos]

        dpos_t = [nav_t - min_ts for nav_t in self._lio_ekf._navs_t]

        # dpos = []
        # for dp in self._lio_ekf._lg_dsp:
        #     if dpos:
        #         dpos.append(dpos[-1] + dp[1])
        #     else:
        #         dpos.append(dp[1])
        dpos = [nav.pos for nav in self._lio_ekf._navs]
        dpos_x = [p[0] for p in dpos]
        dpos_y = [p[1] for p in dpos]
        dpos_z = [p[2] for p in dpos]

        # print("dpos = \n", np.array(dpos))




        scan_t = [scan_begin_ts(ls) - min_ts for ls in self._lio_ekf._lg_scan]

        # fig0, ax_main = plt.subplots(2, 1)


        # Create the plot
        fig, ax = plt.subplots(6, 1, sharex=True, sharey=True)
        for a in ax.flat:
            a.grid(True)


        # ax = plt.figure().add_subplot()  # projection='3d'
        # ax.plot(pos_x, pos_y)  # , pos_z
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')

        # ax.set_zlabel('Z')
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        i = 0
        ax[i].plot(t, acc_x)
        ax[i].set_ylabel('acc_X')
        ax[i + 1].plot(t, acc_y)
        ax[i + 1].set_ylabel('acc_Y')
        ax[i + 2].plot(t, acc_z)
        ax[i + 2].set_ylabel('acc_Z')
        # ax[i + 2].set_xlabel('t')

        # plt.plot(t, acc_x, "r", label="acc_x")
        # plt.grid(True)
        # plt.show()

        # input()


        # Create the plot
        # for a in ax.flat:
        #     a.cla()

        i = 3
        ax[i].plot(t, gyr_x)
        ax[i].set_ylabel('gyr_X')
        ax[i + 1].plot(t, gyr_y)
        ax[i + 1].set_ylabel('gyr_Y')
        ax[i + 2].plot(t, gyr_z)
        ax[i + 2].set_ylabel('gyr_Z')
        ax[i + 2].set_xlabel('t')

        for a in ax:
            a.plot(scan_t, np.zeros_like(scan_t), '8r')

        ax = plt.figure().add_subplot()  # projection='3d'
        plt.axis("equal")
        # ax.plot(pos_x, pos_y)  # , pos_z
        ax.plot(dpos_x, dpos_y)
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # plt.plot(t, acc_x, "r", label="acc_x")
        # plt.grid(True)
        plt.show()

        '''
        import ouster.viz as viz
        from ptudes.utils import (make_point_viz, spin)
        point_viz = make_point_viz(f"Traj: poses = {len(self._lio_ekf._navs)}")

        min_pos = np.zeros(3) if len(
            self._lio_ekf._navs) < 123 else self._lio_ekf._navs[123].pos
        print("min_pos = ", min_pos)
        for idx, nav in enumerate(self._lio_ekf._navs):
            print(f"{idx}: ", nav.pos, ", att_h = ", log_rot_mat(nav.att_h))
            pose_mat = nav.pose_mat()
            pose_mat[:3, 3] -= min_pos
            viz.AxisWithLabel(point_viz, pose=pose_mat,
                              length=0.01)  # label=f"{idx}"

        point_viz.update()
        point_viz.run()
        '''


    def close(self) -> None:
        self._source.close()

    @property
    def metadata(self) -> SensorInfo:
        return self._source.metadata
