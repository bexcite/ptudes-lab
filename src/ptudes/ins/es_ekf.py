from typing import Tuple
from dataclasses import dataclass
import numpy as np
from copy import deepcopy
import scipy
from scipy.spatial.transform import Rotation as R
from ptudes.ins.data import GRAV, IMU, NavState,  set_blk, blk
from ptudes.utils import vee

# TODO: replace with scipy...Rotation
from ouster.sdk.pose_util import exp_rot_vec, log_rot_mat

Vec3 = np.ndarray

DEG2RAD = np.pi / 180.0

UP = np.array([0, 0, 1])
DOWN = np.array([0, 0, -1])

INIT_BACC_STD = 0
INIT_BGYR_STD = 0

@dataclass
class NavErrState:
    dpos: np.ndarray = np.zeros(3)    # Vec3
    datt_v: np.ndarray = np.zeros(3)  # Vec3, tangent-space
    dvel: np.ndarray = np.zeros(3)    # Vec3

    dbias_gyr: np.ndarray = np.zeros(3) # Vec3
    dbias_acc: np.ndarray = np.zeros(3) # Vec3

    dgrav: np.ndarray = np.zeros(3)

    def _formatted_str(self) -> str:
        s = (f"NavStateError:\n"
             f"  dpos: {self.dpos}\n"
             f"  dvel: {self.dvel}\n"
             f"  datt_v: {self.datt_v}\n"
             f"  dbias_gyr: {self.dbias_gyr}\n"
             f"  dbias_acc: {self.dbias_acc}\n"
             f"  dgrav: {self.dgrav}\n")
        return s

    def __repr__(self) -> str:
        return self._formatted_str()


class ESEKF:

    STATE_RANK = 18
    POS_ID = 0
    VEL_ID = 3
    PHI_ID = 6
    BG_ID = 9
    BA_ID = 12
    G_ID = 15

    def __init__(self,
                 *,
                 init_grav: Vec3 = GRAV * DOWN,
                 init_bacc: Vec3 = np.zeros(3),
                 init_bgyr: Vec3 = np.zeros(3)):

        # Initial value of the state
        self._init_pos = np.zeros(3)
        self._init_vel = np.zeros(3)
        self._init_att_v = np.zeros(3)
        self._init_ba = init_bacc
        self._init_bg = init_bgyr
        self._init_grav = init_grav

        self._initpos_std = np.diag([20.0, 20.0, 20.0])
        self._initvel_std = np.diag([5.0, 5.0, 5.0])
        
        initatt_rpy_deg = np.array([15.0, 15.0, 15.5])
        initatt_rotvec = R.from_euler('XYZ', initatt_rpy_deg,
                                      degrees=True).as_rotvec()
        self._initatt_std = np.diag(initatt_rotvec)

        # self._init_bg_std = np.diag([1.0, 1.0, 1.0])
        # self._init_ba_std = np.diag([5.0, 5.0, 5.0])
        self._init_bg_std = np.diag([1.5, 1.5, 1.5])
        self._init_ba_std = np.diag([0.5, 0.5, 0.5])

        self._initg_std = np.diag([2.5, 2.5, 2.5])

        # IMU intrinsics noises
        # self._acc_bias_std = 0.019  # m/s^2 / sqrt(Hz)
        # self._gyr_bias_std = 0.019  # rad/s / sqrt(Hz)
        # self._acc_vrw = 0.0043  # m/s^3 / sqrt(Hz)
        # self._gyr_arw = 0.000266  #  rad/s^2 / sqrt(Hz)

        # TODO: selected for Ouster IMU, some guesses with tests
        self._acc_bias_std = 0.049  # m/s^2 / sqrt(Hz)
        self._gyr_bias_std = 0.38  # rad/s / sqrt(Hz)
        self._acc_vrw = 0.0043  # m/s^3 / sqrt(Hz)
        self._gyr_arw = 0.000466  #  rad/s^2 / sqrt(Hz)

        # print("ESEKF: Imu noise params:")
        # print(f"{self._acc_vrw = }")
        # print(f"{self._gyr_arw = }")
        # print(f"{self._acc_bias_std = }")
        # print(f"{self._gyr_bias_std = }")

        self._imu_corr_time = 3600  # s

        # covariance for error-state Kalman filter
        self._cov = np.zeros((self.STATE_RANK, self.STATE_RANK))
        set_blk(self._cov, self.POS_ID, self.POS_ID,
                np.square(self._initpos_std))
        set_blk(self._cov, self.VEL_ID, self.VEL_ID,
                np.square(self._initvel_std))
        set_blk(self._cov, self.PHI_ID, self.PHI_ID,
                np.square(self._initatt_std))
        set_blk(self._cov, self.BG_ID, self.BG_ID,
                np.square(self._init_bg_std * np.eye(3)))
        set_blk(self._cov, self.BA_ID, self.BA_ID,
                np.square(self._init_ba_std * np.eye(3)))
        set_blk(self._cov, self.G_ID, self.G_ID,
                np.square(self._initg_std * np.eye(3)))

        self._cov_init = np.copy(self._cov)

        # error-state (delta X)
        self._nav_err = NavErrState()
        self._reset_nav_err()

        self._imu_idx = 0

        # make init state
        self._nav_init = NavState()
        self._nav_init.pos = self._init_pos
        self._nav_init.vel = self._init_vel
        self._nav_init.att_v = self._init_att_v
        self._nav_init.bias_gyr = self._init_bg
        self._nav_init.bias_acc = self._init_ba
        # TODO: init grav to ctor?
        self._nav_init.grav = self._init_grav

        self._nav_curr = deepcopy(self._nav_init)
        self._nav_prev = deepcopy(self._nav_curr)

        # imu mechanisation
        self._imu_prev = IMU()
        self._imu_curr = IMU()

        self._imu_initialized = False

        # imu/nav state logging for viz/debugging
        self._lg_t = []
        self._lg_acc = []
        self._lg_gyr = []

        self._navs = []  # nav states (full, after the update on scan)
        self._navs_pred = []  # nav states (after the prediction and before the update)
        self._navs_t = []
        self._nav_update_idxs = []  # idxs to _navs with scans/pose updates


    def processImu(self, imu: IMU) -> None:

        self._imu_prev = self._imu_curr

        imu.dt = imu.ts - self._imu_prev.ts
        # print(f"ESEKF: IMU[{self._imu_idx}] = ", imu)
        self._imu_idx += 1

        self._imu_curr = imu

        if not self._imu_initialized:
            self._imu_initialized = True
            return

        self._nav_prev = deepcopy(self._nav_curr)

        self._insMech()

        dt = self._imu_curr.dt
        acc_body = self._imu_curr.lacc - self._nav_curr.bias_acc

        imu_curr_avel = self._imu_curr.avel - self._nav_curr.bias_gyr
        dtheta = imu_curr_avel * dt
        rot_dtheta = R.from_rotvec(dtheta).as_matrix()

        Fx = np.eye(self.STATE_RANK)
        set_blk(Fx, self.POS_ID, self.VEL_ID, dt * np.eye(3))
        set_blk(Fx, self.VEL_ID, self.PHI_ID, - dt * self._nav_prev.att_h @ vee(acc_body) )
        set_blk(Fx, self.VEL_ID, self.BA_ID, - dt * self._nav_prev.att_h)
        # set_blk(Fx, self.VEL_ID, self.G_ID, dt * np.eye(3))
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

        self._log_on_imu_process()

        # self._nav_prev = deepcopy(self._nav_curr)

    def _insMech(self):
        # compensate bias imu/gyr
        imu_curr_lacc = self._imu_curr.lacc - self._nav_curr.bias_acc
        imu_curr_avel = self._imu_curr.avel - self._nav_curr.bias_gyr

        dt = self._imu_curr.dt

        imu_curr_lacc_g = self._nav_curr.att_h @ imu_curr_lacc
        dtheta = imu_curr_avel * dt
        rot_dtheta = R.from_rotvec(dtheta).as_matrix()

        # predict state
        self._nav_curr.pos = self._nav_curr.pos + self._nav_curr.vel * dt + 0.5 * (
            imu_curr_lacc_g + self._nav_curr.grav) * dt * dt
        self._nav_curr.vel = self._nav_curr.vel + (imu_curr_lacc_g +
                                                   self._nav_curr.grav) * dt
        self._nav_curr.att_h = self._nav_curr.att_h @ rot_dtheta

    def processPose(self, pose_corr: np.ndarray) -> None:

        # logging ...... 
        store_pred = deepcopy(self._nav_curr)
        store_pred.cov = np.copy(self._cov)
        self._navs_pred += [store_pred]

        self._nav_prev = deepcopy(self._nav_curr)

        Rk = self._nav_curr.att_h
        tk = self._nav_curr.pos
        # print("tk = ", tk)
        # print("Rk = ", Rk)

        dR = exp_rot_vec(self._nav_err.datt_v)

        pos = pose_corr[:3, 3]
        rot = pose_corr[:3, :3]

        # add to measurement sets
        Jp = np.zeros((6, self.STATE_RANK))
        set_blk(Jp, 0, self.POS_ID, 1.0 * np.eye(3))
        set_blk(Jp, 3, self.PHI_ID, 1.0 * np.eye(3))
        Epos = np.square(0.02 * np.eye(3))
        Eatt = np.square(0.01 * np.eye(3))
        Epa = scipy.linalg.block_diag(Epos, Eatt)
        resid_pa = np.zeros(6)
        resid_pa[:3] = pos - self._nav_curr.pos - self._nav_err.dpos
        Rk_inv = np.linalg.inv(Rk)
        dR_inv = np.linalg.inv(dR)
        resid_pa[3:] = log_rot_mat(Rk_inv @ dR_inv @ rot)

        S = Jp @ self._cov @ Jp.transpose() + Epa
        K = self._cov @ Jp.transpose() @ np.linalg.inv(S)
        # print("S = \n", S)
        # print("K = \n", K)
        delta_x = K @ resid_pa

        self._cov = (np.eye(self.STATE_RANK) - K @ Jp) @ self._cov

        # self._cov = np.copy(self._cov_init)

        # print("ESEKF: delta_x = \n", delta_x)

        # apply correction error-state
        self._nav_err.dpos += delta_x[self.POS_ID:self.POS_ID + 3]
        self._nav_err.dvel += delta_x[self.VEL_ID:self.VEL_ID + 3]
        self._nav_err.datt_v += delta_x[self.PHI_ID:self.PHI_ID + 3]
        self._nav_err.dbias_gyr += delta_x[self.BG_ID:self.BG_ID + 3]
        self._nav_err.dbias_acc += delta_x[self.BA_ID:self.BA_ID + 3]
        self._nav_err.dgrav += delta_x[self.G_ID:self.G_ID + 3]

        # print(f"ESEKF: _nav_err FINAL [POSE CORR] = \n", self._nav_err)

        # inject error to the current state
        self._nav_curr.pos += self._nav_err.dpos
        self._nav_curr.vel += self._nav_err.dvel
        self._nav_curr.att_h = self._nav_curr.att_h @ exp_rot_vec(self._nav_err.datt_v)
        self._nav_curr.bias_gyr += self._nav_err.dbias_gyr
        self._nav_curr.bias_acc += self._nav_err.dbias_acc
        self._nav_curr.grav += self._nav_err.dgrav

        # covariance projection
        G_theta = np.eye(3) - vee(0.5 * self._nav_err.datt_v)
        phi_block = blk(self._cov, self.PHI_ID, self.PHI_ID, 3)
        set_blk(self._cov, self.PHI_ID, self.PHI_ID, G_theta @ phi_block @ G_theta.transpose())

        # print("\nESEKF_nav_prev (pre POSE CORR UPDATED) = \n", self._nav_prev)
        # print("ESEKF _nav_curr (POSE CORR UPDATED) = \n", self._nav_curr)
        # print("UPDATED COV:::::::::::::::::::::\n", self._cov)

        self._reset_nav_err()

        self._log_on_pose_corr(pose_corr)

        # self._nav_prev = deepcopy(self._nav_curr)

        # input()

    def _reset_nav_err(self):
        self._nav_err.dpos = np.zeros(3)
        self._nav_err.dvel = np.zeros(3)
        self._nav_err.datt_v = np.zeros(3)
        self._nav_err.dbias_gyr = np.zeros(3)
        self._nav_err.dbias_acc = np.zeros(3)
        self._nav_err.dgrav = np.zeros(3)
        # print("ESEKF ------ RESET --- NAV -- ERR-----")

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

    def _log_on_pose_corr(self, pose_corr: np.ndarray):
        store_nav = deepcopy(self._nav_curr)
        store_nav.cov = np.copy(self._cov)
        store_nav.update = True

        # i.e. correction pose
        store_nav.kiss_pose = pose_corr

        self._navs += [store_nav]
        self._navs_t += [self._imu_curr.ts]
        self._nav_update_idxs += [len(self._navs) - 1]
        # print("self._nav_update_idxs = ", self._nav_update_idxs)

        assert len(self._navs) == len(self._navs_pred)