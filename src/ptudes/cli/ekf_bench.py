"""Error-state Extended Kalman Filter experiments"""
from typing import Iterator
import click
from typing import Optional

import ouster.client as client
from ouster.sdk.util import resolve_metadata

from ptudes.utils import (read_metadata_json, read_packet_source)
from ptudes.lio_ekf import LioEkfScans

import numpy as np
import numpy.random as npr
from ptudes.ins.data import IMU, GRAV, ekf_traj_ate
from ptudes.ins.es_ekf import ESEKF
from ptudes.ins.viz_utils import lio_ekf_viz, lio_ekf_graphs, lio_ekf_error_graphs

DOWN = np.array([0, 0, -1])

@click.group(name="ekf-bench")
def ptudes_ekf_bench() -> None:
    """ES EKF benchmarks and experiments.

    Various experiments with Error-state kalman filters and IMU mechanizations.
    """
    pass


def sim_imu(acc_mean: np.ndarray = np.zeros(3),
            acc_std: float = 1.5,
            acc_noise_std: float = 0.4,
            acc_bias: np.ndarray = np.array([0.9, -0.2, -0.4]),
            gyr_mean: np.ndarray = np.zeros(3),
            gyr_std: float = 5.0,
            gyr_noise_std: float = 2.5,
            gyr_bias: np.ndarray = np.array([0.01, 0.03, -0.012]),
            gravity: np.ndarray = GRAV * DOWN,
            freq: float = 100) -> Iterator[IMU]:
    dt = 1 / freq
    imu_idx = 0

    acc_bias = acc_bias
    gyr_bias = gyr_bias

    acc = npr.normal(0.0, acc_std, 3)
    acc = acc_mean + acc - gravity
    gyr = npr.normal(0.0, gyr_std, 3)
    gyr = gyr + gyr_mean

    while True:
        if imu_idx % 10 == 0:
            acc = npr.normal(0.0, acc_std, 3)
            # acc[1] = 0
            # acc[2] = 0
            acc = acc_mean + acc - gravity

            gyr = npr.normal(0.0, gyr_std, 3)
            gyr = gyr + gyr_mean
        # gyr = [0,0,0]
        # gyr[0] = 0
        # gyr[1] = 0
        # gyr[2] = 0
        # gyr = np.zeros(3)
        # add noise and biases
        acc_noise = npr.normal(0, acc_noise_std, 3)
        # acc_noise = np.zeros(3)
        gyr_noise = npr.normal(0.0, gyr_noise_std, 3)
        # gyr_noise = np.zeros(3)
        imu_ideal = IMU(acc, gyr, imu_idx * dt)
        imu_noisy = IMU(acc + acc_noise + acc_bias, gyr + gyr_noise + gyr_bias,
                        imu_idx * dt)
        yield imu_ideal, imu_noisy

        imu_idx += 1


@click.command(name="sim")
@click.option("-t",
              "--duration",
              type=float,
              default=2.0,
              help="Time to generate IMUs measurements (seconds, default 2.0)")
@click.option("-f", "--freq", type=float, default=100.0, help="IMU frequency")
@click.option(
    "--corr-t",
    type=float,
    default=0.1,
    help="Pose correction time interval to EKF (seconds, default 0.1)")
@click.option("--acc-noise-std",
              type=float,
              default=0.4,
              help="IMU accelerometer noise sigma")
@click.option("--gyr-noise-std",
              type=float,
              default=2.4,
              help="IMU gyroscope noise sigma")
@click.option("-p",
              "--plot",
              required=False,
              type=str,
              help="Plotting option [graphs, point_viz]")
def ptudes_ekf_sim(duration: float,
                   corr_t: float,
                   freq: float,
                   acc_noise_std: float,
                   gyr_noise_std: float,
                   plot: Optional[str] = None) -> None:
    """Simulated IMU measurements"""

    ekf_gt = ESEKF()
    ekf = ESEKF()

    print("SIM MEAS")
    initialized = False
    dur_t = duration
    start_ts = 0
    last_corr_t = 0
    for imu_ideal, imu_noisy in sim_imu(freq=freq,
                                        acc_noise_std=acc_noise_std,
                                        gyr_noise_std=gyr_noise_std):
        ts = imu_ideal.ts
        if not initialized:
            start_ts = ts
            last_corr_t = ts
            initialized = True

        print("imu_noisy: ", imu_noisy)
        ekf_gt.processImu(imu_ideal)
        ekf.processImu(imu_noisy)

        if ts - last_corr_t > corr_t:
            ekf.processPose(ekf_gt._nav_curr.pose_mat())
            last_corr_t = ts

        if ts - start_ts > dur_t:
            break

    print("NAV GT:\n", ekf_gt._nav_curr)
    print("NAV:\n", ekf._nav_curr)

    ate_rot, ate_trans = ekf_traj_ate(ekf_gt, ekf)
    print(f"ATE_rot:   {ate_rot:.04f} deg")
    print(f"ATE trans: {ate_trans:.04f} m")


    if not plot:
        return

    if plot == "graphs":
        lio_ekf_graphs(ekf)
        lio_ekf_graphs(ekf_gt)
        lio_ekf_error_graphs(ekf_gt, ekf)
    elif plot == "point_viz":
        lio_ekf_viz(ekf)
    else:
        print(f"WARNING: plot param '{plot}' doesn't "
                f"supported")


ptudes_ekf_bench.add_command(ptudes_ekf_sim)
