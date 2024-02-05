from typing import Optional
from dataclasses import dataclass
import numpy as np
from copy import deepcopy
import scipy
from scipy.spatial.transform import Rotation as R
from ptudes.ins.data import GRAV, IMU, NavState,  set_blk, blk
from ptudes.utils import vee

# TODO: replace with scipy...Rotation?
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

    def reset(self) -> None:
        """Resets the state to zeros"""
        self.dpos = np.zeros(3)
        self.dvel = np.zeros(3)
        self.datt_v = np.zeros(3)
        self.dbias_gyr = np.zeros(3)
        self.dbias_acc = np.zeros(3)
        self.dgrav = np.zeros(3)

    def __repr__(self) -> str:
        return self._formatted_str()


class ESEKF:
    """Error State Extended Kalman Filter for IMU based odometry filters.

    NOTE: It's done to test algos rather than be fast production tool,
          for which you would want to offload ESEKF to the C++ with
          corresponding bindings to Python.
    """

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
                 init_bgyr: Vec3 = np.zeros(3),
                 _logging: bool = False):
        """Creates ES EKF

        TODO: Expose more init params to the ctor
        
        Args:
          init_grav: initial gravity vector direction
          init_bacc: initial accelerometer bias
          init_bgyr: initial gyroscope bias
          _logging: if True, turns on saving the past imus, nav states and
                    covariances for later graphs
        """

        # Initial value of the state
        self._init_pos = np.zeros(3)
        self._init_vel = np.zeros(3)
        self._init_att_v = np.zeros(3)
        self._init_ba = init_bacc
        self._init_bg = init_bgyr
        self._init_grav = init_grav

        self._logging = _logging

        self._initpos_std = np.diag([10.0, 10.0, 10.0])
        self._initvel_std = np.diag([5.0, 5.0, 5.0])

        initatt_rpy_deg = np.array([10.0, 10.0, 10.0])
        initatt_rotvec = R.from_euler('XYZ', initatt_rpy_deg,
                                      degrees=True).as_rotvec()
        self._initatt_std = np.diag(initatt_rotvec)

        self._init_bg_std = np.diag([1.5, 1.5, 1.5])
        self._init_ba_std = np.diag([0.5, 0.5, 0.5])

        self._initg_std = np.diag([2.5, 2.5, 2.5])

        # IMU intrinsics noises
        # TODO: selected for Ouster IMU, some guesses with tests, buut ...
        self._acc_bias_std = 0.049  # m/s^2 / sqrt(Hz)
        self._gyr_bias_std = 0.38  # rad/s / sqrt(Hz)
        self._acc_vrw = 0.0043  # m/s^3 / sqrt(Hz)
        self._gyr_arw = 0.000466  #  rad/s^2 / sqrt(Hz)

        # not used here, but in some ES EKF formulation and IMU mechs its needed
        # self._imu_corr_time = 3600  # s

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

        # state transition derivatives at current estimate
        self._Fx = np.eye(self.STATE_RANK)

        # process and imu measerements noise
        self._W = np.zeros((self.STATE_RANK, self.STATE_RANK))

        # error-state (delta X)
        self._nav_err = NavErrState()

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

    @property
    def nav(self) -> NavState:
        """Get current NavState of the filter"""
        return self._nav_curr

    @property
    def ts(self) -> float:
        """Get current (last) processed timestamp"""
        return self._imu_curr.ts

    def processImu(self, imu: IMU) -> None:
        """EKF predict step using the new imu measurement"""

        self._imu_prev = self._imu_curr

        imu.dt = imu.ts - self._imu_prev.ts
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

        set_blk(self._Fx, self.POS_ID, self.VEL_ID, dt * np.eye(3))
        set_blk(self._Fx, self.VEL_ID, self.PHI_ID, - dt * self._nav_prev.att_h @ vee(acc_body) )
        set_blk(self._Fx, self.VEL_ID, self.BA_ID, - dt * self._nav_prev.att_h)
        # commented out because it's often pulls acc biases, so ...
        # TODO: implement gravity params in an S2 space
        # set_blk(self._Fx, self.VEL_ID, self.G_ID, dt * np.eye(3))
        set_blk(self._Fx, self.PHI_ID, self.PHI_ID, rot_dtheta.transpose())
        set_blk(self._Fx, self.PHI_ID, self.BG_ID, - dt * np.eye(3))

        # prepare process noises
        set_blk(self._W, self.VEL_ID, self.VEL_ID,
                dt * dt * np.square(self._acc_bias_std * np.eye(3)))
        set_blk(self._W, self.PHI_ID, self.PHI_ID,
                dt * dt * np.square(self._gyr_bias_std * np.eye(3)))
        set_blk(self._W, self.BA_ID, self.BA_ID,
                dt * np.square(self._acc_vrw * np.eye(3)))
        set_blk(self._W, self.BG_ID, self.BG_ID,
                dt * np.square(self._gyr_arw * np.eye(3)))

        self._cov = self._Fx @ self._cov @ self._Fx.transpose() + self._W

        self._log_on_imu_process()

    def _insMech(self):
        """IMU mechanization to propagate NavState based on the new IMU meas"""

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

    def processPose(self,
                    pose_corr: np.ndarray,
                    meas_cov: Optional[np.ndarray] = None) -> None:
        """Filter update using pose measurement
        
        Args:
          pose_corr: 4x4 measred pose (i.e. KissICP or some other source)
          meas_cov: 6x6 measurement covariance (i.e. measurement uncertainty)
        """

        # logging ......
        if self._logging:
            store_pred = deepcopy(self._nav_curr)
            store_pred.cov = np.copy(self._cov)
            self._navs_pred += [store_pred]

        self._nav_prev = deepcopy(self._nav_curr)

        Rk = self._nav_curr.att_h
        tk = self._nav_curr.pos

        dR = exp_rot_vec(self._nav_err.datt_v)

        pos = pose_corr[:3, 3]
        rot = pose_corr[:3, :3]

        # add to measurement sets
        Jp = np.zeros((6, self.STATE_RANK))
        set_blk(Jp, 0, self.POS_ID, 1.0 * np.eye(3))
        set_blk(Jp, 3, self.PHI_ID, 1.0 * np.eye(3))
        if meas_cov is None:
            Epos = np.square(0.02 * np.eye(3))
            Eatt = np.square(0.01 * np.eye(3))
            meas_cov = scipy.linalg.block_diag(Epos, Eatt)
        resid_pa = np.zeros(6)
        resid_pa[:3] = pos - self._nav_curr.pos - self._nav_err.dpos
        Rk_inv = np.linalg.inv(Rk)
        dR_inv = np.linalg.inv(dR)
        resid_pa[3:] = log_rot_mat(dR_inv @ Rk_inv @ rot)

        S = Jp @ self._cov @ Jp.transpose() + meas_cov
        K = self._cov @ Jp.transpose() @ np.linalg.inv(S)
        delta_x = K @ resid_pa

        self._cov = (np.eye(self.STATE_RANK) - K @ Jp) @ self._cov

        # apply correction to error-state
        self._nav_err.dpos += delta_x[self.POS_ID:self.POS_ID + 3]
        self._nav_err.dvel += delta_x[self.VEL_ID:self.VEL_ID + 3]
        self._nav_err.datt_v += delta_x[self.PHI_ID:self.PHI_ID + 3]
        self._nav_err.dbias_gyr += delta_x[self.BG_ID:self.BG_ID + 3]
        self._nav_err.dbias_acc += delta_x[self.BA_ID:self.BA_ID + 3]
        self._nav_err.dgrav += delta_x[self.G_ID:self.G_ID + 3]

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

        # reset error state after the update step
        self._nav_err.reset()

        self._log_on_pose_corr(pose_corr)

    def _log_on_imu_process(self):
        """Logging IMU data for graphs"""

        if not self._logging:
            return

        self._lg_t += [self._imu_curr.ts]
        self._lg_acc += [self._imu_curr.lacc.copy()]
        self._lg_gyr += [self._imu_curr.avel.copy()]

        self._navs += [deepcopy(self._nav_curr)]
        self._navs_t += [self._imu_curr.ts]

        store_pred = deepcopy(self._nav_curr)
        store_pred.cov = np.copy(self._cov)
        self._navs_pred += [store_pred]

    def _log_on_pose_corr(self, pose_corr: np.ndarray):
        """Logging pose correstion step for graphs/debugs"""

        if not self._logging:
            return

        store_nav = deepcopy(self._nav_curr)
        store_nav.cov = np.copy(self._cov)
        store_nav.update = True

        # i.e. correction pose
        store_nav.kiss_pose = pose_corr

        self._navs += [store_nav]
        self._navs_t += [self._imu_curr.ts]
        self._nav_update_idxs += [len(self._navs) - 1]

        assert len(self._navs) == len(self._navs_pred)
