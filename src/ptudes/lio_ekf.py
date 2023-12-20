from typing import Optional, Iterable, Union, List, Dict

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

RED_COLOR = np.array([1.0, 0.1, 0.1, 1.0])  # RGBA
BLUE_COLOR = np.array([0.4, 0.4, 1.0, 1.0])  # RGBA
YELLOW_COLOR = np.array([0.1, 1.0, 1.0, 1.0])  # RGBA

GRAV = 9.782940329221166
DEG2RAD = np.pi / 180.0

from .imu_test import imu_raw

def scan_begin_ts(ls: client.LidarScan) -> float:
    return client.first_valid_column_ts(ls) / 10**9

def scan_end_ts(ls: client.LidarScan) -> float:
    return client.last_valid_column_ts(ls) / 10**9

def scan_mid_ts(ls: client.LidarScan) -> float:
    # TODO: not very good for scans with missed packets
    #       and/or not full cloumns windows setting
    bts = scan_begin_ts(ls)
    return bts + 0.5 * (scan_end_ts(ls) - bts)

def centroid(points: np.ndarray) -> np.ndarray:
    return np.sum(points, axis=0) / points.shape[0]

@dataclass
class NavState:
    pos: np.ndarray = np.zeros(3)    # Vec3
    att_h: np.ndarray = np.eye(3)    # Mat3x3, SO(3)
    vel: np.ndarray = np.zeros(3)    # Vec3

    bias_gyr: np.ndarray = np.zeros(3)  # Vec3
    bias_acc: np.ndarray = np.zeros(3) # Vec3

    scan: Optional[LidarScan] = None

    frame: Optional[np.ndarray] = None
    frame_ds: Optional[np.ndarray] = None
    source: Optional[np.ndarray] = None

    src: Optional[np.ndarray] = None
    src_hl: Optional[np.ndarray] = None
    src_source: Optional[np.ndarray] = None
    src_source_hl: Optional[np.ndarray] = None
    tgt: Optional[np.ndarray] = None
    tgt_hl: Optional[np.ndarray] = None

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


@dataclass
class NavErrState:
    dpos: np.ndarray = np.zeros(3)    # Vec3
    datt_v: np.ndarray = np.zeros(3)  # Vec3, tangent-space
    dvel: np.ndarray = np.zeros(3)    # Vec3

    dbias_gyr: np.ndarray = np.zeros(3) # Vec3
    dbias_acc: np.ndarray = np.zeros(3) # Vec3

    def _formatted_str(self) -> str:
        s = (f"NavStateError:\n"
             f"  dpos: {self.dpos}\n"
             f"  dvel: {self.dvel}\n"
             f"  datt_v: {self.datt_v}\n"
             f"  dbias_gyr: {self.dbias_gyr}\n"
             f"  dbias_acc: {self.dbias_acc}\n")
        return s

    def __repr__(self) -> str:
        return self._formatted_str()


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

        np.set_printoptions(precision=3, linewidth=180)

        self._metadata = metadata

        imu_to_sensor = self._metadata.imu_to_sensor_transform.copy()
        imu_to_sensor[:3, 3] /= 1000
        self._sensor_to_imu = np.linalg.inv(imu_to_sensor)

        self._metadata.extrinsic = self._sensor_to_imu
        self._xyz_lut = client.XYZLut(self._metadata, use_extrinsics=True)

        # self._imu_intr_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self._imu_intr_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        self._g_fn = GRAV * np.array([0, 0, -1])

        self._initpos_std = np.diag([10.05, 10.05, 10.05])
        self._initvel_std = np.diag([5.05, 5.05, 1.05])
        self._initatt_std = DEG2RAD * np.diag([15.0, 15.0, 15.5])

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

        self._scan_meas_std = 0.04  # 2 cm (OS-0 on 100% refl target)

        # covariance for error-state Kalman filter
        self._cov = np.zeros((self.STATE_RANK, self.STATE_RANK))
        set_blk(self._cov, self.POS_ID, self.POS_ID,
                np.square(self._initpos_std))
        set_blk(self._cov, self.VEL_ID, self.VEL_ID,
                np.square(self._initvel_std))
        set_blk(self._cov, self.PHI_ID, self.PHI_ID,
                np.square(self._initatt_std))
        set_blk(self._cov, self.BG_ID, self.BG_ID,
                np.square(10* self._gyr_bias_std * np.eye(3)))
        set_blk(self._cov, self.BA_ID, self.BA_ID,
                np.square(10* self._acc_bias_std * np.eye(3)))

        self._cov_init = np.copy(self._cov)

        self._cov_imu_bnoise = np.zeros((self.STATE_RANK, self.STATE_RANK))
        corr_coef = 2 / self._imu_corr_time
        set_blk(self._cov_imu_bnoise, self.BG_ID, self.BG_ID,
                corr_coef * np.square(10* self._gyr_bias_std * np.eye(3)))
        set_blk(self._cov_imu_bnoise, self.BA_ID, self.BA_ID,
                corr_coef * np.square(10* self._acc_bias_std * np.eye(3)))

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
        self._lg_biasa = []
        self._lg_biasg = []


        self._lg_scan = []
        # self._lg_dsp = []

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

        # print(f"delta_grav = {delta_vgrav}")
        # print(f"delta_v_fn  = {delta_v_fn}")

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

        # print("nav_err = \n", self._nav_err)

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

        # np.set_printoptions(precision=3, linewidth=180)
        # print("UPDATED COV:::::::::::::::::::::\n", self._cov)


        # logging
        self._lg_t += [self._imu_curr.ts]
        self._lg_acc += [self._imu_curr.lacc]
        self._lg_gyr += [self._imu_curr.avel]

        self._nav_prev = deepcopy(self._nav_curr)



        # test only when processLidarScan is disabled
        # if not self._navs:
        #     self._reset_nav()
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

        # debug
        kiss_deskew_and_guess = False

        # deskew
        frame = self.deskew_scan(xyz,
                                 timestamps,
                                 linear_prediction=kiss_deskew_and_guess)

        source, frame_downsample = self.voxelize(frame)
        print("source.shape = ", source.shape)
        print("frame_downsample.shape = ", frame_downsample.shape)


        sigma = self.get_sigma_threshold()
        sigma = 2.5
        print("sigma = ", sigma)
        kernel = sigma / 3
        # update state

        # move source to map frams using imu integration of meas between
        # prev and current lidar scan
        Rk = self._nav_curr.att_h
        tk = self._nav_curr.pos
        # exp_datt = exp_rot_vec(self._nav_err.datt_v)
        # if src.size:
        print("tk = ", tk)
        print("Rk = ", Rk)
        print("dtk = ", self._nav_err.dpos)
        # print("exp_datt = ", exp_datt)

        initial_guess = self._nav_curr.pose_mat()
        if kiss_deskew_and_guess:
            initial_guess = self.kiss_initial_guess()
        print("initial_guess = \n", initial_guess)

        src = np.empty((0, 3))
        src_hl = np.empty((0, 3))
        src_source = np.empty((0, 3))
        src_source_hl = np.empty((0, 3))
        tgt = np.empty((0, 3))
        tgt_hl = np.empty((0, 3))

        for it in range(1):
            print(f"--- ITERATION[{it}] =====================:::::")

            if self._local_map.empty():
                break

            dR = exp_rot_vec(self._nav_err.datt_v)

            new_icp_pose = register_frame(
                points=source,
                voxel_map=self._local_map,
                initial_guess=initial_guess,
                max_correspondance_distance=3 * sigma,
                kernel=sigma / 3,
            )
            h_dx_icp = np.matmul(
                new_icp_pose[:3, :3],
                source.transpose()).transpose() + new_icp_pose[:3, 3]

            h_dx = np.matmul(
                dR @ Rk, source.transpose()).transpose() + tk + self._nav_err.dpos
            # print("============================= hdx = ", h_dx.shape)
            # print("source[10] = ", source[:10])
            # print("h_dx[10] = ", h_dx[:10])

            # np.set_printoptions(precision=3, linewidth=180)
            print("PREDICTION COV:::::::::::::::::::::\n", self._cov)

            print("_local_map.size = ", self._local_map.point_cloud().shape)
            print("_local_map_empty = ", self._local_map.empty())

            src, tgt = self._local_map.get_correspondences(h_dx, 3 * sigma)
            print("src = ", src.shape)
            print("tgt = ", tgt.shape)

            src_icp, tgt_icp = self._local_map.get_correspondences(h_dx_icp, 3 * sigma)
            print("src_icp = ", src_icp.shape)
            print("tgt_icp = ", tgt_icp.shape)

            # HACK: recover original coords
            src_source = src - tk - self._nav_err.dpos
            src_source = np.matmul(src_source, np.linalg.inv(dR @ Rk).transpose())

            # HACK: recover original coords
            src_source_icp = src_icp - new_icp_pose[:3, 3]
            src_source_icp = np.matmul(
                src_source_icp,
                np.linalg.inv(new_icp_pose[:3, :3]).transpose())


            # perfect (kiss-icp guided) correspondance src, tgt and src_source
            src_p = (
                np.matmul(dR @ Rk, src_source_icp.transpose()).transpose() +
                tk + self._nav_err.dpos)
            print("centroid src_p = ", centroid(src_p))
            print("centroid tgt_p = ", centroid(tgt_icp))
            print("centroid src - tgt p = ", centroid(src_p) - centroid(tgt_icp))
            src = src_p
            src_source = src_source_icp
            tgt = tgt_icp


            # if src.size:
            #     print("src_10 = ", src[:10])
            #     print("tgt_10 = ", tgt[:10])
            #     print("src_source_10 = ", src_source[:10])
            #     print("resid_10 = ", np.linalg.norm(src - tgt, axis=1)[:10])

            resid = src - tgt

            print("centroid src = ", centroid(src))
            print("centroid tgt = ", centroid(tgt))
            print("centroid src  - tgt = ", centroid(src) - centroid(tgt))

            # self._cov_scan_meas_inv = np.eye(3)
            print("Ep_inv = ", self._cov_scan_meas_inv)

            Ji = np.zeros((3, self.STATE_RANK))
            sum_Ji = np.zeros((self.STATE_RANK, self.STATE_RANK))
            set_blk(Ji, 0, 0, -1.0 * np.eye(3))
            print("Ji init = \n", Ji)
            sum_res = np.zeros(self.STATE_RANK)
            # rand_idx = np.random.randint(0, src.shape[0], 10) if src.shape[0] > 0 else []
            # rand_idx = 2967
            # print("rand_idx = ", rand_idx)
            # idxs = [rand_idx] if src.shape[0] > 0 else []
            idxs = list(range(src.shape[0]))

            # keep only some points for residuals (bigger, etc)
            # print("resid_shape = ", resid.shape)
            # resid_norms = np.linalg.norm(resid, axis=1)
            # print("resid_norms_shape = ", resid_norms.shape)
            # bigger_resid = np.nonzero((resid_norms > 1) & (resid_norms < 3)
            #                           & (src_source[:, 2] > 0)
            #                           & (src_source[:, 1] < -10))
            # print("bigger_resids = ", bigger_resid)
            # print("bigger_resids_size = ", bigger_resid[0].size)
            # idxs = bigger_resid[0]
            # input()

            src_hl = src[idxs]
            tgt_hl = tgt[idxs]
            src_source_hl = src_source[idxs]
            for i in idxs:
                set_blk(Ji, 0, self.PHI_ID, vee(1.0 * Rk @ src_source[i]))
                # print(f"src_source[{i}] = \n", src_source[i])
                # print(f"Ji[{i}] = \n", Ji)
                # print(f"resid[{i}] =", resid[i])
                # input()
                # w = np.square(kernel) / np.square(kernel + np.linalg.norm(resid[i]))
                # print(f"w[{i}] = ", w)
                # w_mat = w * np.eye(3)
                sum_Ji += Ji.transpose() @ self._cov_scan_meas_inv @ Ji
                sum_res += Ji.transpose() @ self._cov_scan_meas_inv @ resid[i]
                # print("ooooo i = ", i)
                # print("sum_res = \n", sum_res)

            print("sum_Ji = \n", sum_Ji)
            print("sum_res = \n", sum_res)
            # print("sum_res.shape = ", sum_res.shape)

            # delta_xx = np.linalg.lstsq(sum_Ji, sum_res, rcond=None)
            # print("delta_xx = ", delta_xx)

            # self._cov = np.copy(self._cov_init)

            cov_inv = np.linalg.inv(self._cov)
            print("cov_inv = \n", cov_inv)
            H_plus_cov = np.linalg.inv(sum_Ji + cov_inv)
            print("H_plus_cov = \n", H_plus_cov)
            # K_gain = H_plus_cov @ Ji.transpose() @ self._cov_scan_meas_inv
            # print("K_gain = \n", K_gain)
            delta_x = H_plus_cov @ sum_res

            print("delta_x = ", delta_x)

            print("H_plus_cov @ sum_Ji = \n", H_plus_cov @ sum_Ji)

            # apply correction error-state
            self._nav_err.dpos += delta_x[self.POS_ID:self.POS_ID + 3]
            self._nav_err.dvel += delta_x[self.VEL_ID:self.VEL_ID + 3]
            self._nav_err.datt_v += delta_x[self.PHI_ID:self.PHI_ID + 3]
            self._nav_err.dbias_gyr += delta_x[self.BG_ID:self.BG_ID + 3]
            self._nav_err.dbias_acc += delta_x[self.BA_ID:self.BA_ID + 3]

            self._cov = (np.eye(self.STATE_RANK) - H_plus_cov @ sum_Ji) @ self._cov

            print(f"_nav_err FINAL [iter:{it}] = \n", self._nav_err)

            print("_nav_curr FINAL (ICP) = ", new_icp_pose)

            # self._nav_curr.pos = new_icp_pose[:3, 3]
            # self._nav_curr.att_h = new_icp_pose[:3, :3]




        self._nav_curr.pos += self._nav_err.dpos
        # self._nav_curr.pos = new_icp_pose[:3, 3]
        self._nav_curr.vel += self._nav_err.dvel
        self._nav_curr.att_h = exp_rot_vec(self._nav_err.datt_v) @ self._nav_curr.att_h
        # self._nav_curr.att_h = new_icp_pose[:3, :3]
        self._nav_curr.bias_gyr += self._nav_err.dbias_gyr
        self._nav_curr.bias_acc += self._nav_err.dbias_acc


        print("\n_nav_prev (pre SCAN UDATED) = \n", self._nav_prev)
        print("_nav_curr (SCAN UDATED) = \n", self._nav_curr)
        print("UPDATED COV:::::::::::::::::::::\n", self._cov)

        new_pose = self._nav_curr.pose_mat()

        self._adaptive_threshold.update_model_deviation(
            np.linalg.inv(initial_guess) @ new_pose)

        self._reset_nav_err()


        # if not self._navs:
        #     print("RRRRRRRRRRRRRRRRRRRREEEEEEEEEEEEERSSSSSSS")
        #     self._reset_nav()

        if self._local_map.empty():
            # first scna processed
            self._reset_nav()
            self._navs = []
            self._navs_t = []


        store_nav = deepcopy(self._nav_curr)
        store_nav.scan = ls
        store_nav.frame = frame
        store_nav.frame_ds = frame_downsample
        store_nav.source = source
        store_nav.src = src
        store_nav.src_hl = src_hl
        store_nav.src_source = src_source
        store_nav.src_source_hl = src_source_hl
        store_nav.tgt = tgt
        store_nav.tgt_hl = tgt_hl
        store_nav.local_map = self._local_map.point_cloud()


        self._local_map.update(frame_downsample, self._nav_curr.pose_mat())


        self._navs += [store_nav]
        self._navs_t += [self._imu_curr.ts]

        # print("navs = \n", self._navs[-3:])
        # print("navs_t = \n", self._navs_t[-3:])

        print("local_map.size = ", self._local_map.point_cloud().shape)
        # print("src.shape = ", src.shape)

        # exit(0)

        self._nav_prev = deepcopy(self._nav_curr)

        # print("_nav_prev = \n", self._nav_prev)

        # input()

        self._lg_scan += [ls]
        # self._lg_dsp += [(scan_begin_ts(ls), self._nav_curr.pos)]
        # self._reset_nav()


    def deskew_scan(self, xyz: np.ndarray,
                    timestamps: np.ndarray,
                    linear_prediction: bool=False) -> np.ndarray:
        if len(self._navs) < 1:
            return xyz
        scan_navs = self._get_scan_nav_idxs()

        last_scan_nav = 0
        if len(scan_navs) == 0:
            if not (len(self._navs) > 8 and len(self._navs) < 15):
                return xyz
        else:
            last_scan_nav = scan_navs[-1]

        print("LAST_SCAN_NAV = ", last_scan_nav)

        if linear_prediction and len(scan_navs) < 2:
            return xyz
        
        to_pose = self._nav_curr.pose_mat()
        if linear_prediction:
            to_pose = self.kiss_initial_guess()

        deskew_frame = kiss_icp_pybind._deskew_scan(
            frame=kiss_icp_pybind._Vector3dVector(xyz),
            timestamps=timestamps,
            start_pose=self._navs[last_scan_nav].pose_mat(),
            finish_pose=to_pose,
        )
        return np.asarray(deskew_frame)

    def get_kiss_prediction_model(self):
        scan_navs = self._get_scan_nav_idxs()
        if len(scan_navs) < 2:
            return np.eye(4)

        nav_pre = self._navs[scan_navs[-2]]
        nav_last = self._navs[scan_navs[-1]]
        return np.linalg.inv(nav_pre.pose_mat()) @ nav_last.pose_mat()
    
    def kiss_initial_guess(self):
        scan_navs = self._get_scan_nav_idxs()
        if len(scan_navs) < 1:
            return np.eye(4)
        if len(scan_navs) < 2:
            return self._navs[scan_navs[0]].pose_mat()
        nav_pre = self._navs[scan_navs[-2]]
        nav_last = self._navs[scan_navs[-1]]
        pred = np.linalg.inv(nav_pre.pose_mat()) @ nav_last.pose_mat()
        return nav_last.pose_mat() @ pred

    def _get_scan_nav_idxs(self) -> List[int]:
        return [i for i, nav in enumerate(self._navs) if nav.scan is not None]

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

        print("start_scan = ", self._start_scan)
        print("end_scan = ", self._end_scan)

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
                        print("NAV_CURR (Ls) = ", self._lio_ekf._nav_curr)

                    yield ls_write

                    if (self._end_scan is not None
                            and scan_idx >= self._end_scan):
                        print("BREAK on scan_idx = ", scan_idx)
                        break

                    scan_idx += 1

                    ls_write = None

            elif isinstance(packet, client.ImuPacket):
                if self._start_scan == 0 or scan_idx >= self._start_scan:

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

                    print("NAV_CURR (Im) = ", self._lio_ekf._nav_curr)

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

        ba_x = [nav.bias_acc[0] for nav in self._lio_ekf._navs]
        ba_y = [nav.bias_acc[1] for nav in self._lio_ekf._navs]
        ba_z = [nav.bias_acc[2] for nav in self._lio_ekf._navs]

        bg_x = [nav.bias_gyr[0] for nav in self._lio_ekf._navs]
        bg_y = [nav.bias_gyr[1] for nav in self._lio_ekf._navs]
        bg_z = [nav.bias_gyr[2] for nav in self._lio_ekf._navs]

        nav_t = [nav_t - min_ts for nav_t in self._lio_ekf._navs_t]

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

        scan_t = [scan_begin_ts(ls) - min_ts for ls in self._lio_ekf._lg_scan]

        # fig0, ax_main = plt.subplots(2, 1)

        # Create the plot
        fig = plt.figure()
        # fig, ax_all = plt.subplots(6, 2, sharex=True, sharey=True)
        ax = [
            plt.subplot(6, 2, 1),
            plt.subplot(6, 2, 3),
            plt.subplot(6, 2, 5),
            plt.subplot(6, 2, 7),
            plt.subplot(6, 2, 9),
            plt.subplot(6, 2, 11)
        ]
        for a in ax[1:]:
            a.sharex(ax[0])
        for a in ax[1:3]:
            a.sharey(ax[0])
        for a in ax[4:]:
            a.sharey(ax[3])
        for a in ax[:-1]:
            plt.setp(a.get_xticklines(), visible=False)
            plt.setp(a.get_xticklabels(), visible=False)
        for a in ax:
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
        ax[i].plot(nav_t, ba_x)
        ax[i].set_ylabel('acc_X')
        ax[i + 1].plot(t, acc_y)
        ax[i + 1].plot(nav_t, ba_y)
        ax[i + 1].set_ylabel('acc_Y')
        ax[i + 2].plot(t, acc_z)
        ax[i + 2].plot(nav_t, ba_z)
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
        ax[i].plot(nav_t, bg_x)
        ax[i].set_ylabel('gyr_X')
        ax[i + 1].plot(t, gyr_y)
        ax[i + 1].plot(nav_t, bg_y)
        ax[i + 1].set_ylabel('gyr_Y')
        ax[i + 2].plot(t, gyr_z)
        ax[i + 2].plot(nav_t, bg_z)
        ax[i + 2].set_ylabel('gyr_Z')
        ax[i + 2].set_xlabel('t')

        # i = 6
        axX = fig.add_subplot(6, 2, (2, 4))
        axX.plot(nav_t, dpos_x)
        axX.grid(True)
        axY = fig.add_subplot(6, 2, (6, 8))
        axY.plot(nav_t, dpos_y)
        axY.grid(True)
        axZ = fig.add_subplot(6, 2, (10, 12))
        axZ.plot(nav_t, dpos_z)
        axZ.grid(True)
        axZ.set_xlabel('t')

        for a in ax + [axX, axY, axZ]:
            a.plot(scan_t, np.zeros_like(scan_t), '8r')

        # plt.autoscale(tight=True)

        # plt.tight_layout()

        '''
        ax = plt.figure().add_subplot()  # projection='3d'
        plt.axis("equal")
        # ax.plot(pos_x, pos_y)  # , pos_z
        ax.plot(dpos_x, dpos_y)
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        '''

        mplot = False

        # plt.plot(t, acc_x, "r", label="acc_x")
        # plt.grid(True)
        if mplot:
            plt.show()
            return



        import ouster.viz as viz
        from ptudes.utils import (make_point_viz, spin)
        from ptudes.viz_utils import PointCloud
        point_viz = make_point_viz(f"Traj: poses = {len(self._lio_ekf._navs)}")

        def next_scan_based_nav(navs: List[NavState],
                                start_idx: int = 0) -> int:
            ln = len(navs)
            start_idx = (start_idx + ln) % ln
            curr_idx = start_idx
            while navs[curr_idx].scan is None:
                curr_idx = (curr_idx + 1) % ln
                if curr_idx == start_idx:
                    break
            return curr_idx

        @dataclass
        class CloudsStruct:
            src_source: PointCloud
            src_source_hl: PointCloud
            src: PointCloud
            tgt: PointCloud
            src_hl: PointCloud
            tgt_hl: PointCloud
            local_map: PointCloud

            def __init__(self, point_viz: viz.PointViz):
                self.src_source = PointCloud(point_viz)
                self.src_source_hl = PointCloud(point_viz,
                                                point_color=YELLOW_COLOR,
                                                point_size=5)
                self.src = PointCloud(point_viz)
                self.src_hl = PointCloud(point_viz,
                                         point_color=RED_COLOR,
                                         point_size=5)
                self.tgt = PointCloud(point_viz)
                self.tgt_hl = PointCloud(point_viz,
                                         point_color=BLUE_COLOR,
                                         point_size=5)
                self.local_map = PointCloud(point_viz)

            def toggle(self):
                self.src_source.toggle()
                self.src_source_hl.toggle()
                self.src.toggle()
                self.src_hl.toggle()
                self.tgt.toggle()
                self.tgt_hl.toggle()
                self.local_map.toggle()

        clouds: Dict[int, CloudsStruct] = dict()

        def set_cloud_from_idx(idx: int):
            nonlocal clouds
            nav = self._lio_ekf._navs[idx]
            if nav.scan is None:
                return
            if idx not in clouds:
                clouds[idx] = CloudsStruct(point_viz)
                clouds[idx].src_source.pose = nav.pose_mat()
                clouds[idx].src_source.points = nav.src_source
                clouds[idx].src_source_hl.pose = nav.pose_mat()
                clouds[idx].src_source_hl.points = nav.src_source_hl
                clouds[idx].src.points = nav.src
                clouds[idx].src_hl.points = nav.src_hl
                clouds[idx].tgt.points = nav.tgt
                clouds[idx].tgt_hl.points = nav.tgt_hl
                clouds[idx].local_map.points = nav.local_map
            print("SET POINTS SIZE.src = ", clouds[idx].src.points.shape)

        def toggle_cloud_from_idx(idx: int, atr: Optional[str] = None):
            nonlocal clouds
            if idx not in clouds:
                return
            if atr is None or not hasattr(clouds[idx], atr):
                clouds[idx].toggle()
            elif hasattr(clouds[idx], atr):
                cloud = getattr(clouds[idx], atr)
                print(f"toggle {atr}[{idx}], size = ", cloud.points.shape)
                cloud.toggle()


        target_idx = next_scan_based_nav(self._lio_ekf._navs, start_idx=0)

        set_cloud_from_idx(target_idx)

        def handle_keys(ctx, key, mods) -> bool:
            nonlocal target_idx
            if key == 32:
                target_idx = next_scan_based_nav(self._lio_ekf._navs,
                                                 start_idx=target_idx + 1)
                target_nav = self._lio_ekf._navs[target_idx]
                print("TNAV: ", target_nav)
                point_viz.camera.set_target(
                    np.linalg.inv(target_nav.pose_mat()))
                set_cloud_from_idx(target_idx)
                point_viz.update()
            elif key == ord('V'):
                toggle_cloud_from_idx(target_idx)
                point_viz.update()
            elif key == ord('G'):
                toggle_cloud_from_idx(target_idx, "src_source")
                toggle_cloud_from_idx(target_idx, "src_source_hl")
                point_viz.update()
            elif key == ord('H'):
                toggle_cloud_from_idx(target_idx, "src")
                toggle_cloud_from_idx(target_idx, "src_hl")
                point_viz.update()
            elif key == ord('J'):
                toggle_cloud_from_idx(target_idx, "tgt")
                toggle_cloud_from_idx(target_idx, "tgt_hl")
                point_viz.update()
            elif key == ord('M'):
                toggle_cloud_from_idx(target_idx, "local_map")
                point_viz.update()
            return True

        point_viz.push_key_handler(handle_keys)

        # min_pos = np.zeros(3) if len(
        #     self._lio_ekf._navs) < 123 else self._lio_ekf._navs[123].pos
        # print("min_pos = ", min_pos)
        for idx, nav in enumerate(self._lio_ekf._navs):
            # print(f"{idx}: ", nav.pos, ", att_h = ", log_rot_mat(nav.att_h))
            pose_mat = nav.pose_mat()
            # pose_mat[:3, 3] -= min_pos
            axis_length = 0.5 if nav.scan is not None else 0.1
            axis_label = f"{idx}" if nav.scan is not None else ""
            viz.AxisWithLabel(point_viz, pose=pose_mat,
                              length=axis_length,
                              label=axis_label)


        target_nav = self._lio_ekf._navs[target_idx]
        point_viz.camera.set_target(np.linalg.inv(target_nav.pose_mat()))

        point_viz.update()
        point_viz.run()




    def close(self) -> None:
        self._source.close()

    @property
    def metadata(self) -> SensorInfo:
        return self._source.metadata
