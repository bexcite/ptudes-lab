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
from ptudes.ins.data import IMU, GRAV, calc_ate, ekf_traj_ate
from ptudes.ins.es_ekf import ESEKF
from ptudes.ins.viz_utils import lio_ekf_viz, lio_ekf_graphs, lio_ekf_error_graphs

DOWN = np.array([0, 0, -1])
UP = np.array([0, 0, 1])

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

    initialized = False
    dur_t = duration
    ts = 0
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

        ekf_gt.processImu(imu_ideal)
        ekf.processImu(imu_noisy)

        if ts - last_corr_t > corr_t:
            ekf.processPose(ekf_gt._nav_curr.pose_mat())
            last_corr_t = ts

        if ts - start_ts > dur_t:
            break

    print(f"processed duration: {ts - start_ts:0.04} s")
    print(f"updates num: {len(ekf._nav_scan_idxs)}\n")

    print("NAV GT:\n", ekf_gt._nav_curr)
    print("NAV:\n", ekf._nav_curr)

    ate_rot, ate_trans = ekf_traj_ate(ekf_gt, ekf)
    print(f"ATE_rot:   {ate_rot:.04f} deg")
    print(f"ATE trans: {ate_trans:.04f} m")

    if plot == "graphs":
        lio_ekf_graphs(ekf)
        lio_ekf_graphs(ekf_gt)
        lio_ekf_error_graphs(ekf_gt, ekf)
    elif plot == "point_viz":
        lio_ekf_viz(ekf)
    elif not plot:
        return
    else:
        print(f"WARNING: plot param '{plot}' doesn't "
                f"supported")


@click.command(name="nc")
@click.argument("file", required=True, type=click.Path(exists=True))
@click.option(
    "-m",
    "--meta",
    required=False,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Metadata for BAG, required if automatic metadata resolution fails")
@click.option("-g",
              "--gt-file",
              required=False,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="Ground truth file with poses to compare")
@click.option("-t",
              "--duration",
              type=float,
              default=2.0,
              help="Time duration of the data read/processed :"
              "(seconds, default 2.0)")
@click.option("--start-ts",
              type=float,
              default=0.0,
              help="Start time (relative to the beginning of the data) "
              "(seconds, default 0.0)")
@click.option("-p",
              "--plot",
              required=False,
              type=str,
              help="Plotting option [graphs, point_viz]")
def ptudes_ekf_nc(file: str,
                  meta: Optional[str] = None,
                  gt_file: Optional[str] = None,
                  duration: float = 2.0,
                  start_ts: float = 0.0,
                  plot: Optional[str] = None) -> None:
    """Newer College 2021 runs"""

    from ptudes.bag import IMUBagSource
    from ptudes.utils import read_newer_college_gt

    imu_source = IMUBagSource(file, imu_topic="/alphasense_driver_ros/imu")

    if not gt_file:
        print("need gt now")
        return

    gts = read_newer_college_gt(gt_file)
    gt_pose0 = gts[0][1]
    pose_corr_idx = 0

    ekf = ESEKF(init_grav=GRAV * UP)

    gt_poses = []

    initialized = False
    dur_t = duration
    ts = 0
    first_ts = 0
    for imu in imu_source:
        ts = imu.ts
        if not initialized:
            first_ts = ts
            initialized = True

        # skipping till the beginning (--start-ts)
        if ts - first_ts < start_ts:
            while pose_corr_idx < len(gts) and ts >= gts[pose_corr_idx][0]:
                print("next gt_ts = ", gts[pose_corr_idx][0])
                pose_corr_idx += 1
            continue

        ekf.processImu(imu)

        if ts >= gts[pose_corr_idx][0]:
            pose_corr = np.linalg.inv(gt_pose0) @ gts[pose_corr_idx][1]

            ekf.processPose(pose_corr)

            gt_poses.append(pose_corr)
            if pose_corr_idx + 1 < len(gts):
                pose_corr_idx += 1

        if ts - first_ts - start_ts > dur_t:
            break

    print("LAST GT POSE:\n", gt_poses[-1])
    print("NAV:\n", ekf._nav_curr)

    navs = [ekf._navs[i] for i in ekf._nav_scan_idxs]

    print(f"scanned duration: {ts - first_ts:0.04} s")
    print(f"updates num: {len(navs)}\n")
    if navs:
        ate_rot, ate_trans = calc_ate(navs, gt_poses)
        print(f"ATE_rot:   {ate_rot:.04f} deg")
        print(f"ATE trans: {ate_trans:.04f} m")

    if not ekf._navs:
        return

    if plot == "graphs":
        lio_ekf_graphs(ekf)
        # lio_ekf_graphs(ekf_gt)
        # lio_ekf_error_graphs(ekf_gt, ekf)
    elif plot == "point_viz":
        lio_ekf_viz(ekf)
    elif not plot:
        return
    else:
        print(f"WARNING: plot param '{plot}' doesn't "
              f"supported")


ptudes_ekf_bench.add_command(ptudes_ekf_sim)
ptudes_ekf_bench.add_command(ptudes_ekf_nc)
