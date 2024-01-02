from typing import Optional, Iterable, Union, List, Dict

import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from copy import deepcopy

from ouster.client import (SensorInfo, LidarScan, ChanField, FieldTypes,
                           UDPProfileLidar)
import ouster.client._client as _client
import ouster.client as client

from ouster.sdk.pose_util import exp_rot_vec, log_rot_mat, _no_scipy_log_rot_mat

from kiss_icp.config import load_config
from kiss_icp.pybind import kiss_icp_pybind
from kiss_icp.voxelization import voxel_down_sample
from kiss_icp.mapping import get_voxel_hash_map
from kiss_icp.threshold import get_threshold_estimator
from kiss_icp.registration import register_frame

import matplotlib.pyplot as plt

from ptudes.kiss import KissICPWrapper
from ptudes.data import GRAV, IMU, NavState, set_blk, blk

from ptudes.utils import read_newer_college_gt

from ptudes.viz_utils import lio_ekf_graphs, lio_ekf_error_graphs, lio_ekf_viz

from ptudes.viz_utils import (RED_COLOR, BLUE_COLOR, YELLOW_COLOR, GREY_COLOR,
                              GREY_COLOR1, WHITE_COLOR)

from dataclasses import dataclass

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
        imu_to_sensor[:3, 3] /= 1000  # mm to m convertion
        self._sensor_to_imu = np.linalg.inv(imu_to_sensor)

        # exploiting extrinsics mechanics of Ouster SDK to
        # make an XYZLut that transforms lidar points to the
        #  Nav (imu) frame
        self._metadata.extrinsic = self._sensor_to_imu
        self._xyz_lut = client.XYZLut(self._metadata, use_extrinsics=True)

        # self._imu_intr_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self._imu_intr_rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # self._g_fn = GRAV * np.array([0, 0, -1])
        self._g_fn = GRAV * np.array([0, 0, 1])
        # self._g_fn = GRAV * np.array([-0.00054619, 9.77265772, -0.81460648])

        self._initpos_std = np.diag([20.05, 20.05, 20.05])
        self._initvel_std = np.diag([5.05, 5.05, 5.05])
        self._initatt_std = DEG2RAD * np.diag([45.0, 45.0, 45.5])
        # self._initbg_std = np.diag([1.0, 1.0, 1.0])
        # self._initba_std = np.diag([5.0, 5.0, 5.0])
        self._initbg_std = np.diag([1.5, 1.5, 1.5])
        self._initba_std = np.diag([0.5, 0.5, 0.5])


        self._initba = np.zeros(3)
        self._initbg = np.zeros(3)
        # self._initba = np.array([-0.04, -0.12, 0.2])
        # self._initbg = np.array([-0.004, -0.001, 0.0035])

        # super good (from IMU IAM-20680HT Datasheet)
        # self._acc_vrw = 50 * 1e-3 * GRAV  # m/s^2
        # self._gyr_arw = 1 * DEG2RAD  #  rad/s
        # self._acc_bias_std = 135 * 1e-6 * GRAV  # m/s^2 / sqrt(Hz)
        # self._gyr_bias_std = 0.005 * DEG2RAD  # rad/s / sqrt(Hz)

        # self._acc_vrw = 50 * 1e-3 * GRAV  # m/s^2
        # self._gyr_arw = 10 * DEG2RAD  #  rad/s
        # self._acc_bias_std = 135 * 1e-6 * GRAV  # m/s^2 / sqrt(H)
        # self._gyr_bias_std = 0.01 * DEG2RAD  # rad/s / sqrt(H)

        # IMU0 (Newer College)
        self._acc_vrw = 0.0043  # m/s^2
        self._gyr_arw = 0.000266  #  rad/s
        self._acc_bias_std = 0.019  # m/s^2 / sqrt(Hz)
        self._gyr_bias_std = 0.019  # rad/s / sqrt(Hz)

        print("Imu noise params:")
        print(f"{self._acc_vrw = }")
        print(f"{self._gyr_arw = }")
        print(f"{self._acc_bias_std = }")
        print(f"{self._gyr_bias_std = }")

        # exit(0)

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
        # set_blk(self._cov, self.BG_ID, self.BG_ID,
        #         np.square(5000 * self._gyr_bias_std * np.eye(3)))
        # set_blk(self._cov, self.BA_ID, self.BA_ID,
        #         np.square(5000 * self._acc_bias_std * np.eye(3)))
        set_blk(self._cov, self.BG_ID, self.BG_ID,
                np.square(self._initbg_std * np.eye(3)))
        set_blk(self._cov, self.BA_ID, self.BA_ID,
                np.square(self._initba_std * np.eye(3)))

        self._cov_init = np.copy(self._cov)

        self._cov_imu_bnoise = np.zeros((self.STATE_RANK, self.STATE_RANK))
        corr_coef = 2 / self._imu_corr_time
        set_blk(self._cov_imu_bnoise, self.BG_ID, self.BG_ID,
                corr_coef * np.square(self._gyr_bias_std * np.eye(3)))
        set_blk(self._cov_imu_bnoise, self.BA_ID, self.BA_ID,
                corr_coef * np.square(self._acc_bias_std * np.eye(3)))

        self._cov_mnoise = np.zeros((6, 6))
        set_blk(self._cov_mnoise, 0, 0, np.square(self._gyr_arw * np.eye(3)))
        set_blk(self._cov_mnoise, 3, 3, np.square(self._acc_vrw * np.eye(3)))

        # point measurement covariance (Er)
        self._cov_scan_meas = np.square(self._scan_meas_std * np.eye(3))
        self._cov_scan_meas_inv = np.linalg.inv(self._cov_scan_meas)

        # error-state (delta X)
        self._nav_err = NavErrState()
        self._reset_nav_err()

        self._imu_idx = 0
        self._scan_idx = 0

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

        # imu mechanisation
        self._imu_prev = IMU()
        self._imu_curr = IMU()


        # use KISS-ICP as a bootstrapping/initialization source
        self._booting = True
        self._kiss_icp = KissICPWrapper(self._metadata,
                                        _use_extrinsics=True,
                                        _min_range=2,
                                        _max_range=70)

        # kiss_icp_config = load_config(None, deskew=True, max_range = 100)
        # self._kiss_icp = KissICP(config=kiss_icp_config)
        self._local_map = get_voxel_hash_map(self._kiss_icp._config)
        # self._adaptive_threshold = get_threshold_estimator(self._kiss_icp._config)
        # print("kiss_icp = ", self._kiss_icp._config)
        # print("adaptive_threshold = ", self._adaptive_threshold)
        # exit(0)

        self._navs = []  # nav states (full, after the update on scan)
        self._navs_pred = []  # nav states (after the prediction and before the update)
        self._navs_t = []
        self._nav_scan_idxs = []  # idxs to _navs with scans

        self._imu_total = IMU()
        self._imu_total_cnt = 0

        self._imu_initialized = False

        self._lg_t = []
        self._lg_acc = []
        self._lg_gyr = []

        print(f"init: nav_pre  = {self._nav_prev}")
        print(f"init: nav_curr = {self._nav_curr}")

    def _reset_nav(self):
        self._nav_curr.pos = np.zeros(3)
        self._nav_curr.vel = np.zeros(3)
        self._nav_curr.att_h = np.eye(3)
        # self._nav_curr.bias_gyr = np.zeros(3)
        # self._nav_curr.bias_acc = np.zeros(3)
        self._nav_curr.bias_gyr = self._initbg
        self._nav_curr.bias_acc = self._initba
        self._nav_prev = deepcopy(self._nav_curr)
        self._cov = np.copy(self._cov_init)
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

        self._nav_curr.pos = self._nav_prev.pos + 0.5 * sk * (
            self._nav_curr.vel + self._nav_prev.vel)

        rot_vec_b = imucurr_angle + np.cross(imupre_angle, imucurr_angle) / 12
        self._nav_curr.att_h = self._nav_prev.att_h @ exp_rot_vec(rot_vec_b)

        # print("det att_h = ", np.linalg.det(self._nav_curr.att_h))
        # r = self._nav_curr.att_h
        # print("rrt = ", r @ r.transpose())
        # input()

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

        # accumulate totals
        self._imu_total.lacc += self._imu_curr.lacc
        self._imu_total.avel += self._imu_curr.avel
        self._imu_total_cnt += 1

        if not self._imu_initialized:
            self._imu_initialized = True
            return

        self._insMech()

        sk = self._imu_curr.dt

        # print("SK = ", sk)

        # update error-state (delta X)
        # TODO: Use real R(k-1)!
        dk = self._nav_prev.att_h @ np.cross(self._imu_curr.lacc,
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

        # print("OPT1 = \n", self._nav_prev.att_h @ vee(self._imu_curr.lacc))
        # print("OPT2 = \n", vee(self._nav_prev.att_h @ self._imu_curr.lacc))
        set_blk(Ak, self.VEL_ID, self.PHI_ID,
                sk * self._nav_prev.att_h @ vee(self._imu_curr.lacc))
        set_blk(Ak, self.VEL_ID, self.BA_ID, sk * self._nav_prev.att_h)
        set_blk(Ak, self.PHI_ID, self.BG_ID, -1.0 * sk * self._nav_prev.att_h)

        Bk = np.zeros((self.STATE_RANK, 6))
        set_blk(Bk, self.VEL_ID, 3,
                -sk * vee(self._nav_err.datt_v) @ self._nav_prev.att_h)

        self._cov = (Ak @ self._cov @ Ak.transpose() +
                     Bk @ self._cov_mnoise @ Bk.transpose() +
                     sk * self._cov_imu_bnoise)

        # np.set_printoptions(precision=3, linewidth=180)
        # print("UPDATED COV:::::::::::::::::::::\n", self._cov)

        self._log_on_imu_process()

        self._nav_prev = deepcopy(self._nav_curr)

        # self._cov = np.copy(self._cov_init)

    def _log_on_imu_process(self):
        # logging IMU data for graphs
        self._lg_t += [self._imu_curr.ts]
        self._lg_acc += [self._imu_curr.lacc.copy()]
        self._lg_gyr += [self._imu_curr.avel.copy()]


        # test only when processLidarScan is disabled
        # if not self._navs:
        #     self._reset_nav()
        self._navs += [deepcopy(self._nav_curr)]
        self._navs_t += [self._imu_curr.ts]

        store_pred = deepcopy(self._nav_curr)
        store_pred.cov = np.copy(self._cov)
        self._navs_pred += [store_pred]


    def processImuAlt(self, imu: IMU) -> None:

        self._imu_prev = self._imu_curr

        imu.dt = imu.ts - self._imu_prev.ts
        print(f"IMU ALT[{self._imu_idx}] = ", imu)
        self._imu_idx += 1

        self._imu_curr = imu

        if not self._imu_initialized:
            self._imu_initialized = True
            return

        self._insMechAlt()

        dt = self._imu_curr.dt

        acc_body = self._imu_curr.lacc - self._nav_curr.bias_acc

        imu_curr_avel = self._imu_curr.avel - self._nav_curr.bias_gyr
        dtheta = imu_curr_avel * dt
        rot_dtheta = Rotation.from_rotvec(dtheta).as_matrix()

        Fx = np.eye(self.STATE_RANK)
        set_blk(Fx, self.POS_ID, self.VEL_ID, dt * np.eye(3))
        set_blk(Fx, self.VEL_ID, self.PHI_ID, - dt * self._nav_prev.att_h @ vee(acc_body) )
        set_blk(Fx, self.VEL_ID, self.BA_ID, - dt * self._nav_prev.att_h )
        set_blk(Fx, self.PHI_ID, self.PHI_ID, rot_dtheta.transpose())
        set_blk(Fx, self.PHI_ID, self.BG_ID, - dt * np.eye(3))

        W = np.zeros((self.STATE_RANK, self.STATE_RANK))
        set_blk(W, self.VEL_ID, self.VEL_ID,
                dt * dt * np.square(self._acc_bias_std * np.eye(3)))
        set_blk(W, self.PHI_ID, self.PHI_ID,
                dt * dt * np.square(self._gyr_bias_std * np.eye(3)))
        set_blk(W, self.BA_ID, self.BA_ID,
                dt * np.square(self._acc_vrw * np.eye(3)))
        set_blk(W, self.BG_ID, self.BG_ID,
                dt * np.square(self._gyr_arw * np.eye(3)))
        

        self._cov = Fx @ self._cov @ Fx.transpose() + W


        # print("SK = ", sk)

        # update error-state (delta X)
        # # TODO: Use real R(k-1)!
        # dk = self._nav_prev.att_h @ np.cross(self._imu_curr.lacc,
        #                                      self._nav_err.datt_v)
        # self._nav_err.dpos = self._nav_err.dpos + self._nav_err.dvel * sk

        # self._nav_err.dvel = (
        #     self._nav_err.dvel + sk * dk +
        #     sk * self._nav_prev.att_h @ self._nav_err.dbias_acc)

        # self._nav_err.datt_v = (
        #     self._nav_err.datt_v - sk *
        #     (self._nav_prev.att_h @ self._nav_err.dbias_gyr))

        # print("nav_err = \n", self._nav_err)

        # Ak = np.eye(self.STATE_RANK)
        # set_blk(Ak, self.POS_ID, self.VEL_ID, sk * np.eye(3))

        # # print("OPT1 = \n", self._nav_prev.att_h @ vee(self._imu_curr.lacc))
        # # print("OPT2 = \n", vee(self._nav_prev.att_h @ self._imu_curr.lacc))
        # set_blk(Ak, self.VEL_ID, self.PHI_ID,
        #         sk * self._nav_prev.att_h @ vee(self._imu_curr.lacc))
        # set_blk(Ak, self.VEL_ID, self.BA_ID, sk * self._nav_prev.att_h)
        # set_blk(Ak, self.PHI_ID, self.BG_ID, -1.0 * sk * self._nav_prev.att_h)

        # Bk = np.zeros((self.STATE_RANK, 6))
        # set_blk(Bk, self.VEL_ID, 3,
        #         -sk * vee(self._nav_err.datt_v) @ self._nav_prev.att_h)

        # self._cov = (Ak @ self._cov @ Ak.transpose() +
        #              Bk @ self._cov_mnoise @ Bk.transpose() +
        #              sk * self._cov_imu_bnoise)

        # np.set_printoptions(precision=3, linewidth=180)
        # print("UPDATED COV:::::::::::::::::::::\n", self._cov)

        # self._cov = np.copy(self._cov_init)

        self._log_on_imu_process()

        self._nav_prev = deepcopy(self._nav_curr)

        # self._cov = np.copy(self._cov_init)

    def _insMechAlt(self):
        # print(f"nav_prev = {self._nav_prev}")
        # print(f"nav_curr = {self._nav_curr}")

        # print(f"imu_prev = {self._imu_prev}")
        # print(f"imu_curr = {self._imu_curr}")

        # compensate bias imu/gyr
        imu_curr_lacc = self._imu_curr.lacc - self._nav_curr.bias_acc
        imu_curr_avel = self._imu_curr.avel - self._nav_curr.bias_gyr

        dt = self._imu_curr.dt

        imu_curr_lacc_g = self._nav_curr.att_h @ imu_curr_lacc
        dtheta = imu_curr_avel * dt
        rot_dtheta = Rotation.from_rotvec(dtheta).as_matrix()

        self._nav_curr.pos = self._nav_curr.pos + self._nav_curr.vel * dt + 0.5 * (
            imu_curr_lacc_g + self._g_fn) * dt * dt
        self._nav_curr.vel = self._nav_curr.vel + (imu_curr_lacc_g +
                                                   self._g_fn) * dt
        self._nav_curr.att_h = self._nav_curr.att_h @ rot_dtheta


        # imucurr_vel = self._imu_curr.lacc * sk
        # imucurr_angle = self._imu_curr.avel * sk
        # imupre_vel = self._imu_prev.lacc * sk
        # imupre_angle = self._imu_prev.avel * sk

        # p1 = np.cross(imucurr_angle, imucurr_vel) / 2   #  1/2 * (w(k) x a(k) * s(k)^2)
        # p2 = np.cross(imupre_angle, imucurr_vel) / 12   #  1/12 * (w(k-1) x a(k)) * s(k)^2
        # p3 = np.cross(imucurr_angle, imupre_vel) / 12   #  1/12 * (a(k-1) x w(k)) * s(k)^2

        # delta_v_fb = imucurr_vel + p1 + p2 - p3

        # delta_v_fn = self._nav_prev.att_h @ delta_v_fb

        # # gravity vel part
        # delta_vgrav = self._g_fn * sk

        # # print(f"delta_grav = {delta_vgrav}")
        # # print(f"delta_v_fn  = {delta_v_fn}")

        # self._nav_curr.vel = self._nav_prev.vel + delta_vgrav + delta_v_fn

        # self._nav_curr.pos = self._nav_prev.pos + 0.5 * sk * (
        #     self._nav_curr.vel + self._nav_prev.vel)

        # rot_vec_b = imucurr_angle + np.cross(imupre_angle, imucurr_angle) / 12
        # self._nav_curr.att_h = self._nav_prev.att_h @ exp_rot_vec(rot_vec_b)

        # print("det att_h = ", np.linalg.det(self._nav_curr.att_h))
        # r = self._nav_curr.att_h
        # print("rrt = ", r @ r.transpose())
        # input()

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


    def processLidarPacket(self, lidar_packet: client.LidarPacket) -> None:
        col_ts = lidar_packet.timestamp[0]
        # print(f"ts: {col_ts}, local_ts: {local_ts:.6f}, "
        #       f"processLidar: {lidar_packet = }")

    def processLidarScan(self, ls: client.LidarScan) -> None:

        # self._cov = np.copy(self._cov_init)

        store_pred = deepcopy(self._nav_curr)
        store_pred.cov = np.copy(self._cov)
        self._navs_pred += [store_pred]
        # print("self._navs_pred = ", self._navs_pred[-1])
        # input()

        if self._booting:
            self._kiss_icp.register_frame(ls)
            print("BOOTING: .... kiss icp pose:\n", self._kiss_icp.pose)
            print("velocity: ", self._kiss_icp.velocity)

        # get XYZ points
        timestamps = np.tile(np.linspace(0, 1.0, ls.w, endpoint=False), (ls.h, 1))
        # filtering our zero returns makes it substantially faster for kiss-icp
        sel_flag = ls.field(ChanField.RANGE) != 0
        xyz = self._xyz_lut(ls)[sel_flag]
        timestamps = timestamps[sel_flag]
        # print("xyz.shape = ", xyz.shape)
        # print("xyz.flags = ", xyz.flags)
        # print("timestamps.size = ", timestamps.shape)

        # debug
        kiss_deskew = True
        kiss_guess = True

        # deskew
        if self._booting:
            frame = self._kiss_icp.deskew(xyz, timestamps)
        else:
            frame = self.deskew_scan(xyz, timestamps)

        source, frame_downsample = self.voxelize(frame)
        # print("source.shape = ", source.shape)
        # print("frame_downsample.shape = ", frame_downsample.shape)


        # sigma = self.get_sigma_threshold()
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
        if kiss_guess:
            initial_guess = self.kiss_initial_guess()
        # print("initial_guess = \n", initial_guess)

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
            # print("PREDICTION COV:::::::::::::::::::::\n", self._cov)

            # print("_local_map.size = ", self._local_map.point_cloud().shape)
            # print("_local_map_empty = ", self._local_map.empty())

            src, tgt = self._local_map.get_correspondences(h_dx, 3 * sigma)
            # print("src = ", src.shape)
            # print("tgt = ", tgt.shape)

            src_icp, tgt_icp = self._local_map.get_correspondences(h_dx_icp, 3 * sigma)
            # print("src_icp = ", src_icp.shape)
            # print("tgt_icp = ", tgt_icp.shape)

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
            # print("centroid src_p = ", centroid(src_p))
            # print("centroid tgt_p = ", centroid(tgt_icp))
            # print("centroid src - tgt p = ", centroid(src_p) - centroid(tgt_icp))
            src = src_p
            src_source = src_source_icp
            tgt = tgt_icp


            # if src.size:
            #     print("src_10 = ", src[:10])
            #     print("tgt_10 = ", tgt[:10])
            #     print("src_source_10 = ", src_source[:10])
            #     print("resid_10 = ", np.linalg.norm(src - tgt, axis=1)[:10])

            resid = src - tgt

            # print("centroid src = ", centroid(src))
            # print("centroid tgt = ", centroid(tgt))
            # print("centroid src  - tgt = ", centroid(src) - centroid(tgt))

            # self._cov_scan_meas_inv = np.eye(3)
            # print("Ep_inv = ", self._cov_scan_meas_inv)


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

            sum_Ji = np.zeros((self.STATE_RANK, self.STATE_RANK))
            sum_res = np.zeros(self.STATE_RANK)

            if not self._booting:

                Ji = np.zeros((3, self.STATE_RANK))
                set_blk(Ji, 0, self.POS_ID, -1.0 * np.eye(3))
                print("Ji init = \n", Ji)

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

            else:
                # print("KISS_ALL_POSES = ", self._kiss_icp.poses)
                # scan_navs = [self._navs[sn] for sn in self._get_scan_nav_idxs()]

                # print("ALL_NAVS_POSES = ", scan_navs)

                pos = self._kiss_icp.pose[:3, 3]
                rot = self._kiss_icp.pose[:3, :3]

                # add to measurement sets
                Jp = np.zeros((6, self.STATE_RANK))
                set_blk(Jp, 0, self.POS_ID, 1.0 * np.eye(3))
                set_blk(Jp, 3, self.PHI_ID, 1.0 * np.eye(3))
                # set_blk(Jp, 3, self.PHI_ID, rot.transpose())
                Epos = np.square(0.05 * np.eye(3))
                Eatt = np.square(0.1 * np.eye(3))
                Epos_inv = np.linalg.inv(Epos)
                Eatt_inv = np.linalg.inv(Eatt)
                Epa_inv = scipy.linalg.block_diag(Epos_inv, Eatt_inv)
                Epa = scipy.linalg.block_diag(Epos, Eatt)
                # Epa = scipy.linalg.block_diag(Epos_inv)
                # print("Epa = \n", Epa)
                # print("Epa_inv = \n", Epa_inv)
                sum_Ji += Jp.transpose() @ Epa_inv @ Jp
                resid_pa = np.zeros(6)
                resid_pa[:3] = pos - self._nav_curr.pos - self._nav_err.dpos
                Rk_inv = np.linalg.inv(Rk)
                dR_inv = np.linalg.inv(dR)
                resid_pa[3:] = log_rot_mat(rot @ Rk_inv @ dR_inv)
                # print("dR = ", dR)
                # print("Rk = ", Rk)
                # print("rotv = ", log_rot_mat(dR @ Rk @ rot.transpose()))
                # resid_pa[3:] = log_rot_mat(dR @ Rk @ rot.transpose())
                print("resid_pa = ", resid_pa)
                sum_res += Jp.transpose() @ Epa_inv @ resid_pa
                # print("sum_Ji + Pos = \n", sum_Ji)
                # print("sum_res + Pos = \n", sum_res)

                # S = Jp @ self._cov @ Jp.transpose() + Epa
                # K = self._cov @ Jp.transpose() @ np.linalg.inv(S)
                # print("S = \n", S)
                # print("K = \n", K)
                # delta_x = K @ resid_pa




                # if len(self._kiss_icp.poses) >= 2:
                #     # last_pose = self._nav_curr.pose_mat()
                #     # last_pose = new_icp_pose
                #     # prev_pose = self._navs[scan_navs[0]].pose_mat()
                #     # dp = np.linalg.inv(prev_pose) @ last_pose
                #     # dt = self._imu_curr.ts - self._navs_t[scan_navs[0]]
                #     # print("DP = ", dp)
                #     # print("DT = ", dt)
                #     # vel = dp[:3, 3] / dt
                #     vel = self._kiss_icp.velocity
                #     print("VEL = ", vel)
                #     # self._nav_curr.vel = vel

                #     # add to measurement sets
                #     Jv = np.zeros((3, self.STATE_RANK))
                #     set_blk(Jv, 0, self.VEL_ID, 1.0 * np.eye(3))
                #     Ev_inv = np.linalg.inv(np.square(0.001 * np.eye(3)))
                #     sum_Ji += Jv.transpose() @ Ev_inv @ Jv
                #     sum_res += Jv.transpose() @ Ev_inv @ (
                #         vel - self._nav_curr.vel - self._nav_err.dvel)

                #     print("sum_Ji + V = \n", sum_Ji)
                #     print("sum_res + V = \n", sum_res)

            # print("sum_Ji = \n", sum_Ji)
            # print("sum_res = \n", sum_res)

            # HACK update velocity
            # scan_navs = self._get_scan_nav_idxs()
            # if len(scan_navs) == 1:
            #     # last_pose = self._nav_curr.pose_mat()
            #     last_pose = new_icp_pose
            #     prev_pose = self._navs[scan_navs[0]].pose_mat()
            #     dp = np.linalg.inv(prev_pose) @ last_pose
            #     dt = self._imu_curr.ts - self._navs_t[scan_navs[0]]
            #     print("DP = ", dp)
            #     print("DT = ", dt)
            #     vel = dp[:3, 3] / dt
            #     print("VEL = ", vel)
            #     # self._nav_curr.vel = vel

            #     # add to measurement sets
            #     Jv = np.zeros((3, self.STATE_RANK))
            #     set_blk(Jv, 0, self.VEL_ID, 1.0 * np.eye(3))
            #     Ev_inv = np.linalg.inv(0.001 * np.eye(3))
            #     sum_Ji += Jv.transpose() @ Ev_inv @ Jv
            #     sum_res += Jv.transpose() @ Ev_inv @ (
            #         vel - self._nav_curr.vel - self._nav_err.dvel)

            #     print("sum_Ji + V = \n", sum_Ji)
            #     print("sum_res + V = \n", sum_res)
            # print("sum_res.shape = ", sum_res.shape)

            # delta_xx = np.linalg.lstsq(sum_Ji, sum_res, rcond=None)
            # print("delta_xx = ", delta_xx)

            # self._cov = np.copy(self._cov_init)

            cov_inv = np.linalg.inv(self._cov)
            # print("cov_inv = \n", cov_inv)
            H_plus_cov = np.linalg.inv(sum_Ji + cov_inv)
            # print("H_plus_cov = \n", H_plus_cov)
            # K_gain = H_plus_cov @ Ji.transpose() @ self._cov_scan_meas_inv
            # print("K_gain = \n", K_gain)
            # print("delta_x0 = \n", delta_x)
            delta_x = H_plus_cov @ sum_res

            # print("delta_x = \n", delta_x)

            # print("H_plus_cov @ sum_Ji = \n", H_plus_cov @ sum_Ji)

            # apply correction error-state
            self._nav_err.dpos += delta_x[self.POS_ID:self.POS_ID + 3]
            self._nav_err.dvel += delta_x[self.VEL_ID:self.VEL_ID + 3]
            self._nav_err.datt_v += delta_x[self.PHI_ID:self.PHI_ID + 3]
            # self._nav_err.datt_v = log_rot_mat(
            #     exp_rot_vec(delta_x[self.PHI_ID:self.PHI_ID + 3])
            #     @ exp_rot_vec(self._nav_err.datt_v))
            self._nav_err.dbias_gyr += delta_x[self.BG_ID:self.BG_ID + 3]
            self._nav_err.dbias_acc += delta_x[self.BA_ID:self.BA_ID + 3]

            self._cov = (np.eye(self.STATE_RANK) - H_plus_cov @ sum_Ji) @ self._cov

            print(f"_nav_err FINAL [iter:{it}] = \n", self._nav_err)

            # print("_nav_curr FINAL (ICP) = ", new_icp_pose)

            # self._nav_curr.pos = new_icp_pose[:3, 3]
            # self._nav_curr.att_h = new_icp_pose[:3, :3]




        self._nav_curr.pos += self._nav_err.dpos
        # self._nav_curr.pos = new_icp_pose[:3, 3]
        self._nav_curr.vel += self._nav_err.dvel
        self._nav_curr.att_h = exp_rot_vec(self._nav_err.datt_v) @ self._nav_curr.att_h
        # self._nav_curr.att_h =  self._nav_curr.att_h @ exp_rot_vec(self._nav_err.datt_v)
        # self._nav_curr.att_h = new_icp_pose[:3, :3]
        self._nav_curr.bias_gyr += self._nav_err.dbias_gyr
        self._nav_curr.bias_acc += self._nav_err.dbias_acc


        # HACK update velocity
        # scan_navs = self._get_scan_nav_idxs()
        # if len(scan_navs) == 1:
        #     last_pose = self._nav_curr.pose_mat()
        #     prev_pose = self._navs[scan_navs[0]].pose_mat()
        #     dp = np.linalg.inv(prev_pose) @ last_pose
        #     dt = self._imu_curr.ts - self._navs_t[scan_navs[0]]
        #     print("DP = ", dp)
        #     print("DT = ", dt)
        #     vel = dp[:3, 3] / dt
        #     print("VEL = ", vel)
        #     self._nav_curr.vel = vel


        print("\n_nav_prev (pre SCAN UDATED) = \n", self._nav_prev)
        print("_nav_curr (SCAN UDATED) = \n", self._nav_curr)
        print("UPDATED COV:::::::::::::::::::::\n", self._cov)

        new_pose = self._nav_curr.pose_mat()

        # self._adaptive_threshold.update_model_deviation(
        #     np.linalg.inv(initial_guess) @ new_pose)

        self._reset_nav_err()


        # if not self._navs:
        #     print("RRRRRRRRRRRRRRRRRRRREEEEEEEEEEEEERSSSSSSS")
        #     self._reset_nav()

        if self._local_map.empty():
            # first scan processed
            self._reset_nav()
            self._navs = []
            self._navs_t = []
            self._navs_pred = [self._navs_pred[-1]]


        store_nav = deepcopy(self._nav_curr)
        store_nav.cov = np.copy(self._cov)
        store_nav.update = True
        store_nav.scan = ls
        store_nav.xyz = xyz
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

        if self._booting:
            store_nav.kiss_pose = self._kiss_icp.pose
            store_nav.kiss_map = self._kiss_icp.local_map_points


        self._local_map.update(frame_downsample, self._nav_curr.pose_mat())


        self._navs += [store_nav]
        self._navs_t += [self._imu_curr.ts]
        self._nav_scan_idxs += [len(self._navs) - 1]
        print("self._nav_scan_idxs = ", self._nav_scan_idxs)

        assert len(self._navs) == len(self._navs_pred)

        # print("navs = \n", self._navs[-3:])
        # print("navs_t = \n", self._navs_t[-3:])

        print("local_map.size = ", self._local_map.point_cloud().shape)
        # print("src.shape = ", src.shape)

        # exit(0)

        self._nav_prev = deepcopy(self._nav_curr)

        self._scan_idx += 1

        # print("_nav_prev = \n", self._nav_prev)

        # input()


    def processPoseCorrection(self, pose_corr: np.ndarray) -> None:

        # self._cov = np.copy(self._cov_init)

        store_pred = deepcopy(self._nav_curr)
        store_pred.cov = np.copy(self._cov)
        self._navs_pred += [store_pred]

        Rk = self._nav_curr.att_h
        tk = self._nav_curr.pos
        # exp_datt = exp_rot_vec(self._nav_err.datt_v)
        # if src.size:
        print("tk = ", tk)
        print("Rk = ", Rk)

        dR = exp_rot_vec(self._nav_err.datt_v)

        sum_Ji = np.zeros((self.STATE_RANK, self.STATE_RANK))
        sum_res = np.zeros(self.STATE_RANK)

        pos = pose_corr[:3, 3]
        rot = pose_corr[:3, :3]

        # add to measurement sets
        Jp = np.zeros((3, self.STATE_RANK))
        set_blk(Jp, 0, self.POS_ID, 1.0 * np.eye(3))
        # set_blk(Jp, 3, self.PHI_ID, 1.0 * np.eye(3))
        # set_blk(Jp, 3, self.PHI_ID, rot.transpose())
        Epos = np.square(0.05 * np.eye(3))
        Eatt = np.square(0.1 * np.eye(3))
        Epos_inv = np.linalg.inv(Epos)
        Eatt_inv = np.linalg.inv(Eatt)
        # Epa_inv = scipy.linalg.block_diag(Epos_inv, Eatt_inv)
        Epa_inv = scipy.linalg.block_diag(Epos_inv)
        Epa = scipy.linalg.block_diag(Epos, Eatt)
        # Epa = scipy.linalg.block_diag(Epos_inv)
        # print("Epa = \n", Epa)
        # print("Epa_inv = \n", Epa_inv)
        sum_Ji += Jp.transpose() @ Epa_inv @ Jp
        resid_pa = np.zeros(3)
        resid_pa[:3] = pos - self._nav_curr.pos - self._nav_err.dpos
        Rk_inv = np.linalg.inv(Rk)
        dR_inv = np.linalg.inv(dR)
        # resid_pa[3:] = log_rot_mat(rot @ Rk_inv @ dR_inv)
        # print("dR = ", dR)
        # print("Rk = ", Rk)
        # print("rotv = ", log_rot_mat(dR @ Rk @ rot.transpose()))
        # resid_pa[3:] = log_rot_mat(dR @ Rk @ rot.transpose())
        print("resid_pa = ", resid_pa)
        sum_res += Jp.transpose() @ Epa_inv @ resid_pa
        # print("sum_Ji + Pos = \n", sum_Ji)
        # print("sum_res + Pos = \n", sum_res)

        # S = Jp @ self._cov @ Jp.transpose() + Epa
        # K = self._cov @ Jp.transpose() @ np.linalg.inv(S)
        # print("S = \n", S)
        # print("K = \n", K)
        # delta_x = K @ resid_pa

        # self._cov = np.copy(self._cov_init)

        cov_inv = np.linalg.inv(self._cov)
        # print("cov_inv = \n", cov_inv)
        H_plus_cov = np.linalg.inv(sum_Ji + cov_inv)
        # print("H_plus_cov = \n", H_plus_cov)
        # K_gain = H_plus_cov @ Ji.transpose() @ self._cov_scan_meas_inv
        # print("K_gain = \n", K_gain)
        # print("delta_x0 = \n", delta_x)
        delta_x = H_plus_cov @ sum_res

        print("delta_x = \n", delta_x)

        self._cov = (np.eye(self.STATE_RANK) - H_plus_cov @ sum_Ji) @ self._cov

        # print("H_plus_cov @ sum_Ji = \n", H_plus_cov @ sum_Ji)

        # apply correction error-state
        self._nav_err.dpos += delta_x[self.POS_ID:self.POS_ID + 3]
        self._nav_err.dvel += delta_x[self.VEL_ID:self.VEL_ID + 3]
        self._nav_err.datt_v += delta_x[self.PHI_ID:self.PHI_ID + 3]
        # self._nav_err.datt_v = log_rot_mat(
        #     exp_rot_vec(delta_x[self.PHI_ID:self.PHI_ID + 3])
        #     @ exp_rot_vec(self._nav_err.datt_v))
        self._nav_err.dbias_gyr += delta_x[self.BG_ID:self.BG_ID + 3]
        self._nav_err.dbias_acc += delta_x[self.BA_ID:self.BA_ID + 3]

        print(f"_nav_err FINAL [POSE CORR] = \n", self._nav_err)

        self._nav_curr.pos += self._nav_err.dpos
        # self._nav_curr.pos = new_icp_pose[:3, 3]
        self._nav_curr.vel += self._nav_err.dvel
        self._nav_curr.att_h = exp_rot_vec(self._nav_err.datt_v) @ self._nav_curr.att_h
        # self._nav_curr.att_h =  self._nav_curr.att_h @ exp_rot_vec(self._nav_err.datt_v)
        # self._nav_curr.att_h = new_icp_pose[:3, :3]
        self._nav_curr.bias_gyr += self._nav_err.dbias_gyr
        self._nav_curr.bias_acc += self._nav_err.dbias_acc


        print("\n_nav_prev (pre POSE CORR UPDATED) = \n", self._nav_prev)
        print("_nav_curr (POSE CORR UPDATED) = \n", self._nav_curr)
        print("UPDATED COV:::::::::::::::::::::\n", self._cov)

        self._reset_nav_err()

        self._log_on_pose_corr(pose_corr)

        self._nav_prev = deepcopy(self._nav_curr)

        # input()

    def processPoseCorrectionAlt(self, pose_corr: np.ndarray) -> None:

        store_pred = deepcopy(self._nav_curr)
        store_pred.cov = np.copy(self._cov)
        self._navs_pred += [store_pred]

        Rk = self._nav_curr.att_h
        tk = self._nav_curr.pos
        # exp_datt = exp_rot_vec(self._nav_err.datt_v)
        # if src.size:
        print("tk = ", tk)
        print("Rk = ", Rk)

        dR = exp_rot_vec(self._nav_err.datt_v)

        sum_Ji = np.zeros((self.STATE_RANK, self.STATE_RANK))
        sum_res = np.zeros(self.STATE_RANK)

        pos = pose_corr[:3, 3]
        rot = pose_corr[:3, :3]

        # add to measurement sets
        Jp = np.zeros((6, self.STATE_RANK))
        set_blk(Jp, 0, self.POS_ID, 1.0 * np.eye(3))
        set_blk(Jp, 3, self.PHI_ID, 1.0 * np.eye(3))
        # set_blk(Jp, 3, self.PHI_ID, rot.transpose())
        Epos = np.square(0.05 * np.eye(3))
        Eatt = np.square(0.1 * np.eye(3))
        Epos_inv = np.linalg.inv(Epos)
        Eatt_inv = np.linalg.inv(Eatt)
        # Epa_inv = scipy.linalg.block_diag(Epos_inv, Eatt_inv)
        Epa_inv = scipy.linalg.block_diag(Epos_inv, Eatt_inv)
        Epa = scipy.linalg.block_diag(Epos, Eatt)
        # Epa = scipy.linalg.block_diag(Epos_inv)
        # print("Epa = \n", Epa)
        # print("Epa_inv = \n", Epa_inv)
        sum_Ji += Jp.transpose() @ Epa_inv @ Jp
        resid_pa = np.zeros(6)
        resid_pa[:3] = pos - self._nav_curr.pos - self._nav_err.dpos
        Rk_inv = np.linalg.inv(Rk)
        dR_inv = np.linalg.inv(dR)
        resid_pa[3:] = log_rot_mat(Rk_inv @ dR_inv @ rot)
        # print("dR = ", dR)
        # print("Rk = ", Rk)
        # print("rotv = ", log_rot_mat(dR @ Rk @ rot.transpose()))
        # resid_pa[3:] = log_rot_mat(dR @ Rk @ rot.transpose())
        print("resid_pa = ", resid_pa)
        sum_res += Jp.transpose() @ Epa_inv @ resid_pa
        # print("sum_Ji + Pos = \n", sum_Ji)
        # print("sum_res + Pos = \n", sum_res)

        S = Jp @ self._cov @ Jp.transpose() + Epa
        K = self._cov @ Jp.transpose() @ np.linalg.inv(S)
        # print("S = \n", S)
        # print("K = \n", K)
        delta_x = K @ resid_pa

        self._cov = (np.eye(self.STATE_RANK) - K @ Jp) @ self._cov

        # self._cov = np.copy(self._cov_init)

        # cov_inv = np.linalg.inv(self._cov)
        # print("cov_inv = \n", cov_inv)
        # H_plus_cov = np.linalg.inv(sum_Ji + cov_inv)
        # print("H_plus_cov = \n", H_plus_cov)
        # K_gain = H_plus_cov @ Ji.transpose() @ self._cov_scan_meas_inv
        # print("K_gain = \n", K_gain)
        # print("delta_x0 = \n", delta_x)
        # delta_x = H_plus_cov @ sum_res

        print("delta_x = \n", delta_x)

        # self._cov = (np.eye(self.STATE_RANK) - H_plus_cov @ sum_Ji) @ self._cov

        # print("H_plus_cov @ sum_Ji = \n", H_plus_cov @ sum_Ji)

        # apply correction error-state
        self._nav_err.dpos += delta_x[self.POS_ID:self.POS_ID + 3]
        self._nav_err.dvel += delta_x[self.VEL_ID:self.VEL_ID + 3]
        self._nav_err.datt_v += delta_x[self.PHI_ID:self.PHI_ID + 3]
        # self._nav_err.datt_v = log_rot_mat(
        #     exp_rot_vec(delta_x[self.PHI_ID:self.PHI_ID + 3])
        #     @ exp_rot_vec(self._nav_err.datt_v))
        self._nav_err.dbias_gyr += delta_x[self.BG_ID:self.BG_ID + 3]
        self._nav_err.dbias_acc += delta_x[self.BA_ID:self.BA_ID + 3]

        print(f"_nav_err FINAL [POSE CORR] = \n", self._nav_err)

        self._nav_curr.pos += self._nav_err.dpos
        # self._nav_curr.pos = new_icp_pose[:3, 3]
        self._nav_curr.vel += self._nav_err.dvel
        # rot_att = Rotation.from_rotvec(self._nav_err.datt_v) * Rotation.from_quat(self._nav_curr.att_q)
        # # # self._nav_curr.att_h = exp_rot_vec(self._nav_err.datt_v) @ self._nav_curr.att_h
        # q = rot_att.as_quat()
        # self._nav_curr.att_q = q #  / np.linalg.norm(q)
        self._nav_curr.att_h = self._nav_curr.att_h @ exp_rot_vec(self._nav_err.datt_v)
        # self._nav_curr.att_h =  self._nav_curr.att_h @ exp_rot_vec(self._nav_err.datt_v)
        # self._nav_curr.att_h = new_icp_pose[:3, :3]
        self._nav_curr.bias_gyr += self._nav_err.dbias_gyr
        self._nav_curr.bias_acc += self._nav_err.dbias_acc


        G_theta = np.eye(3) - vee(0.5 * self._nav_err.datt_v)
        phi_block = blk(self._cov, self.PHI_ID, self.PHI_ID, 3)
        set_blk(self._cov, self.PHI_ID, self.PHI_ID, G_theta @ phi_block @ G_theta.transpose())


        print("\n_nav_prev (pre POSE CORR UPDATED) = \n", self._nav_prev)
        print("_nav_curr (POSE CORR UPDATED) = \n", self._nav_curr)
        print("UPDATED COV:::::::::::::::::::::\n", self._cov)

        self._reset_nav_err()

        self._log_on_pose_corr(pose_corr)

        self._nav_prev = deepcopy(self._nav_curr)

        # input()

    def _log_on_pose_corr(self, pose_corr: np.ndarray):
        store_nav = deepcopy(self._nav_curr)
        store_nav.cov = np.copy(self._cov)
        store_nav.update = True

        # i.e. correction pose
        store_nav.kiss_pose = pose_corr

        self._navs += [store_nav]
        self._navs_t += [self._imu_curr.ts]
        self._nav_scan_idxs += [len(self._navs) - 1]
        print("self._nav_scan_idxs = ", self._nav_scan_idxs)

        assert len(self._navs) == len(self._navs_pred)

        # self._cov = np.copy(self._cov_init)

    def deskew_scan(self, xyz: np.ndarray,
                    timestamps: np.ndarray,
                    linear_prediction: bool=False) -> np.ndarray:
        if len(self._navs) < 1:
            return xyz
        scan_navs = self._get_scan_nav_idxs()

        last_nav = 0
        if len(scan_navs) == 0:
            if not (len(self._navs) > 8 and len(self._navs) < 15):
                return xyz
        else:
            last_nav = scan_navs[-1]

        print("LAST_NAV = ", last_nav)

        if linear_prediction and len(scan_navs) < 2:
            return xyz

        to_pose = self._nav_curr.pose_mat()
        if linear_prediction:
            to_pose = self._navs[last_nav].pose_mat(
            ) @ self.linear_scan_prediction()

        deskew_frame = kiss_icp_pybind._deskew_scan(
            frame=kiss_icp_pybind._Vector3dVector(xyz),
            timestamps=timestamps,
            start_pose=self._navs[last_nav].pose_mat(),
            finish_pose=to_pose,
        )
        return np.asarray(deskew_frame)

    def linear_scan_prediction(self):
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
        return self._nav_scan_idxs

    def voxelize(self, orig_frame) -> np.ndarray:
        frame_downsample = voxel_down_sample(
            orig_frame, self._kiss_icp._config.mapping.voxel_size * 0.5)
        source = voxel_down_sample(frame_downsample,
                                   self._kiss_icp._config.mapping.voxel_size)
        return source, frame_downsample

    def get_sigma_threshold(self) -> float:
        adaptive = (self._kiss_icp._config.adaptive_threshold.initial_threshold
                    if not self.has_moved() else
                    self._adaptive_threshold.get_threshold())
        print("ADAOTIVE ==== ", adaptive)
        return adaptive

    def has_moved(self):
        if len(self._navs) < 1:
            return False
        compute_motion = lambda T1, T2: np.linalg.norm((np.linalg.inv(T1) @ T2)[:3, -1])
        motion = compute_motion(self._navs[0].pose_mat(), self._navs[-1].pose_mat())
        return motion > 5 * self._kiss_icp._config.adaptive_threshold.min_motion_th


class LioEkfScans(client.ScanSource):
    """LIO EKF experimental implementation"""

    def __init__(self, source: client.PacketSource,
                 *,
                 fields: Optional[FieldTypes] = None,
                 _start_scan: Optional[int] = None,
                 _end_scan: Optional[int] = None,
                 _plotting: Optional[str] = None) -> None:
        self._source = source
        self._plotting = _plotting

        self._start_scan = _start_scan or 0
        self._end_scan = _end_scan

        self._fields: Union[FieldTypes, UDPProfileLidar]
        self._fields = (fields if fields is not None else
                        self._source.metadata.format.udp_profile_lidar)

        self._lio_ekf = LioEkf(source.metadata)

        self._lio_ekf_gt = LioEkf(source.metadata)
        self._lio_ekf_corr = LioEkf(source.metadata)


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

        bag_file = "/home/pavlo/data/newer-college/2021-ouster-os0-128-alphasense/collection 1 - newer college/2021-07-01-10-37-38-quad-easy.bag"
        from ptudes.bag import IMUBagSource
        imu_source = IMUBagSource(bag_file, imu_topic="/alphasense_driver_ros/imu")
        gts = read_newer_college_gt("/home/pavlo/data/newer-college/2021-ouster-os0-128-alphasense/collection 1 - newer college/ground_truth/gt-nc-quad-easy.csv")
        gt_ts = gts[0][0]
        gt_pose = gts[0][1]
        # for idx, im in enumerate(imus):
        #     print(f"dt[{idx}] = ", gts[0][0] - im.ts)
        pose_idx = 1
        imu_it = iter(imu_source)
        pose_corr = np.linalg.inv(gt_pose) @ gts[pose_idx][1]
        pose_ts = gts[pose_idx][0]

        # print("dts = ", imus[0].ts, gts[0][0], gts[0][0] - imus[0].ts)
        print("gt_pose = ", gt_pose)

        imu = IMU()
        imu_noisy = IMU()
        import numpy.random as npr

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
                    # if scan_idx >= self._start_scan:
                    #     self._lio_ekf.processLidarScan(ls_write)
                    #     print("NAV_CURR (Ls) = ", self._lio_ekf._nav_curr)

                    # yield ls_write

                    if (self._end_scan is not None
                            and scan_idx >= self._end_scan):
                        print("BREAK on scan_idx = ", scan_idx)
                        break

                    scan_idx += 1

                    ls_write = None

            elif isinstance(packet, client.ImuPacket):
                if self._start_scan == 0 or scan_idx >= self._start_scan:

                    # acc_x = 0.1 if imu_idx < 100 else 0
                    # acc_y = 0.0 if imu_idx < 10 else 0.1
                    # acc_x = 2.0
                    # acc_y = 0.0

                    # acc = [acc_x, acc_y, GRAV]
                    # gyr = [0, 0, 5.0]
                    # imu = IMU(acc, gyr, imu_idx * 0.01)

                    # imu_r = imu_raw[imu_idx, :]
                    # acc = imu_r[:3] * GRAV
                    # gyr = imu_r[3:]
                    # imu = IMU(acc, gyr, imu_idx * 0.001)

                    # self._lio_ekf.processImu(imu)

                    # self._lio_ekf.processImuPacket(packet)

                    imu = next(imu_it)


                    # if imu_idx % 10 == 0:
                    #     acc = npr.normal(0.0, 1.0, 3)
                    #     # acc[1] = 0
                    #     # acc[2] = 0
                    #     acc = acc - self._lio_ekf._g_fn
                    #     gyr = npr.normal(0.0, 1.0, 3)
                    #     # gyr = [0,0,0]
                    #     # gyr[0] = 0
                    #     # gyr[1] = 0
                    #     # gyr[2] = 0
                    #     # gyr = np.zeros(3)
                    #     imu = IMU(acc, gyr, imu_idx * 0.01)
                    #     imu_noisy = IMU(acc + np.array([0.9, -0.2, -0.4]), gyr + np.array([0.01, 0.03, -0.012]), imu_idx * 0.01)
                    #     print("IMU 0: ", imu)
                    #     print("IMU 1: ", imu_noisy)
                    # else:
                    #     imu.ts = imu_idx * 0.01
                    #     imu_noisy.ts = imu_idx * 0.01
                    #     # input()

                    if imu.ts > pose_ts:
                        print("MAKE CORRECTION FOR GT! pose_ts = ", pose_ts)
                        print("pose_corr = \n", pose_corr)
                        self._lio_ekf.processPoseCorrectionAlt(pose_corr)
                        pose_idx += 1
                        pose_corr = np.linalg.inv(gt_pose) @ gts[pose_idx][1]
                        pose_ts = gts[pose_idx][0]

                    self._lio_ekf.processImuAlt(deepcopy(imu))

                    # self._lio_ekf.processImuAlt(deepcopy(imu_noisy))
                    # self._lio_ekf_corr.processImuAlt(deepcopy(imu_noisy))

                    # print("imu_to_gt = ", imu)
                    # self._lio_ekf_gt.processImuAlt(deepcopy(imu))

                    # if (imu_idx + 1) % 10 == 0:
                    #     # print(f"NAV_CURR_GT[{imu_idx}] = ", self._lio_ekf_gt._nav_curr)
                    #     # print(f"NAV_CURR[{imu_idx}] = ", self._lio_ekf._nav_curr)
                    #     # print(f"NAV_CURR CORR[{imu_idx}] = ", self._lio_ekf_corr._nav_curr)
                    #     self._lio_ekf_corr.processPoseCorrectionAlt(
                    #         self._lio_ekf_gt._nav_curr.pose_mat())
                    #     # print(f"0NAV_CURR_GT[{imu_idx}] = ", self._lio_ekf_gt._nav_curr)
                    #     # print(f"0NAV_CURR[{imu_idx}] = ", self._lio_ekf._nav_curr)
                    #     # print(f"0NAV_CURR CORR[{imu_idx}] = ", self._lio_ekf_corr._nav_curr)
                    #     # input()

                    # print(f"GYR xyz[len: {imu_idx}]:\n")
                    # for i in range(len(self._lio_ekf._lg_gyr)):
                    #     print(self._lio_ekf._lg_gyr[i])
                    # input()

                    print("NAV_CURR (Im) = ", self._lio_ekf._nav_curr)

                imu_idx += 1

            # if imu_cnt > 150:
            #     break

        print(f"Finished: imu_idx = {imu_idx}, "
              f"scan_idx = {scan_idx}, "
              f"scans_num = {self._lio_ekf._scan_idx}")

        print("NAV_CURR_GT = ", self._lio_ekf_gt._nav_curr)
        print("NAV_CURR = ", self._lio_ekf._nav_curr)
        print("NAV_CURR CORR = ", self._lio_ekf_corr._nav_curr)

        if self._plotting == "graphs":
            lio_ekf_graphs(self._lio_ekf)
            # lio_ekf_error_graphs(self._lio_ekf_gt, self._lio_ekf_corr, self._lio_ekf)

            # lio_ekf_error_graphs(self._lio_ekf_gt, self._lio_ekf)
            # lio_ekf_error_graphs(self._lio_ekf_gt, self._lio_ekf_corr)
            # lio_ekf_graphs(self._lio_ekf_gt)
            # lio_ekf_graphs(self._lio_ekf)
        elif self._plotting == "point_viz":
            # lio_ekf_viz(self._lio_ekf_gt)
            lio_ekf_viz(self._lio_ekf)
            # lio_ekf_viz(self._lio_ekf_corr)
        else:
            print(f"WARNING: plotting param '{self._plotting}' doesn't "
                  f"supported")



    def close(self) -> None:
        self._source.close()

    @property
    def metadata(self) -> SensorInfo:
        return self._source.metadata
