"""Error-state Extended Kalman Filter experiments"""
from typing import Iterator, List
from datetime import datetime
import os
import time
import click
from typing import Optional

import ouster.client as client
from ouster.client import ChanField
from ouster.sdk.util import resolve_metadata
import ouster.sdk.pose_util as pu

from ptudes.utils import (read_metadata_json, read_packet_source,
                          read_newer_college_gt, save_poses_kitti_format,
                          save_poses_nc_gt_format, filter_nc_gt_by_close_ts,
                          reduce_active_beams, filter_nc_gt_by_cmp)

import numpy as np
import numpy.random as npr
from scipy.spatial.transform import Rotation
from ptudes.ins.data import (IMU, GRAV, calc_ate, ekf_traj_ate,
                             _collect_navs_from_gt, StreamStatsTracker)
from ptudes.ins.es_ekf import ESEKF
from ptudes.ins.viz_utils import (ekf_viz, ekf_graphs,
                                  ekf_error_graphs, gt_poses_graphs)
from ptudes.data import OusterLidarData, last_valid_packet_ts
from ptudes.kiss import KissICPWrapper
from tqdm import tqdm
from itertools import cycle

DOWN = np.array([0, 0, -1])
UP = np.array([0, 0, 1])

@click.group(name="ekf-bench")
def ptudes_ekf_bench() -> None:
    """ES EKF benchmarks and experiments.

    Experiments with Error-state kalman filters and IMU mechanizations.
    """
    pass


def sim_imu(acc_mean: np.ndarray = np.zeros(3),
            acc_std: float = 1.5,
            acc_noise_std: float = 0.4,
            acc_bias: np.ndarray = np.array([0.9, -0.2, -0.4]),
            gyr_mean: np.ndarray = np.zeros(3),
            gyr_std: float = 1.0,
            gyr_noise_std: float = 0.2,
            gyr_bias: np.ndarray = np.array([0.01, 0.03, -0.012]),
            gravity: np.ndarray = GRAV * DOWN,
            freq: float = 100) -> Iterator[IMU]:
    dt = 1 / freq
    imu_idx = 0

    acc = npr.normal(0.0, acc_std, 3)
    acc = acc + acc_mean - gravity

    gyr = npr.normal(0.0, gyr_std, 3)
    gyr = gyr + gyr_mean

    while True:
        if imu_idx % 10 == 0:
            acc = npr.normal(0.0, acc_std, 3)
            acc = acc + acc_mean - gravity
            gyr = npr.normal(0.0, gyr_std, 3)
            gyr = gyr + gyr_mean
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
              default=0.4,
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
    """EKF with simulated IMU measurements.

    Noise/bias free integrated measurements used as ground truth for pose
    corrections (ekf updates).
    """

    print("Using sim IMUs with params:")
    print(f"  freq: {freq} Hz")
    print(f"  acc_noise_std: {acc_noise_std}")
    print(f"  gyr_noise_std: {gyr_noise_std}")
    print(f"  correction dt: {corr_t:.02} s")

    print("Running EKF ... \n")

    ekf_gt = ESEKF(_logging=True)
    ekf = ESEKF(_logging=True)

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
            ekf.processPose(ekf_gt.nav.pose_mat())
            last_corr_t = ts

        if ts - start_ts > dur_t:
            break

    print("Results:")

    print(f"processed duration: {ts - start_ts:0.04} s")
    print(f"updates num: {len(ekf._nav_update_idxs)}\n")

    print("NAV GT:\n", ekf_gt.nav)
    print("NAV:\n", ekf.nav)

    ate_rot, ate_trans = ekf_traj_ate(ekf_gt, ekf)
    print(f"ATE_rot:   {ate_rot:.04f} deg")
    print(f"ATE trans: {ate_trans:.04f} m")

    # get associated nav states with gt states and time
    gt_t, gt_navs, navs = _collect_navs_from_gt(ekf_gt, ekf)
    gt_poses = [nav.pose_mat() for nav in gt_navs]

    if plot == "graphs":
        ekf_graphs(ekf, gt=(gt_t, gt_poses))
        ekf_error_graphs(ekf_gt, ekf)
    elif plot == "point_viz":
        ekf_viz(ekf)
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
              required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="Ground truth file with poses to compare and correct poses")
@click.option("-t",
              "--duration",
              type=float,
              default=0.0,
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
@click.option('--xy-plot',
              is_flag=True,
              help="Draw X and Y dimenstions on XY plane, instead of separate")
@click.option("-i",
              "--imu-topic",
              required=False,
              default="/os_node/imu_packets",
              type=str,
              help="Imu topic name to use (msg/Imu or imu_packets)")
def ptudes_ekf_nc(file: str,
                  meta: Optional[str] = None,
                  gt_file: Optional[str] = None,
                  duration: float = 2.0,
                  start_ts: float = 0.0,
                  plot: Optional[str] = None,
                  xy_plot: bool = False,
                  imu_topic: Optional[str] = None) -> None:
    """EKF with Newer College Dataset IMUs topics.

    Ground truth (--gt-file) is used for pose correction.
    """

    from ptudes.bag import IMUBagSource

    # TODO: handle properly known extrinsics calibration of NC datasets
    init_grav = GRAV * UP
    # NOTE: IMUs have different nav frames (ouster vs alphasense) ...
    if imu_topic in ["/os_cloud_node/imu", "/os_node/imu_packets"]:
        init_grav = GRAV * DOWN
    print("init_grav = ", init_grav)

    print("Reading NC dataset:")
    print(f"  file: {file}")
    print(f"  topic: {imu_topic}")
    print(f"  gt file: {gt_file}")

    imu_source = IMUBagSource(file, imu_topic=imu_topic)

    if not gt_file:
        print("need gt now")
        return

    gts = read_newer_college_gt(gt_file)

    pose_corr_idx = 0
    gt_pose0 = np.linalg.inv(gts[pose_corr_idx][1])

    print("Running EKF ... \n")

    ekf = ESEKF(init_grav=init_grav, _logging=bool(plot))

    gt_t = []
    gt_poses = []

    res_poses = []

    gt0_initialized = False
    dur_t = duration
    ts = 0
    first_ts = -1
    for imu in imu_source:
        ts = imu.ts
        if first_ts < 0:
            first_ts = ts

        # skipping till the beginning (--start-ts)
        if ts - first_ts < start_ts:
            continue

        if not gt0_initialized:
            while pose_corr_idx < len(gts) and ts >= gts[pose_corr_idx][0]:
                pose_corr_idx += 1
            gt_pose0 = np.linalg.inv(gts[pose_corr_idx][1])
            gt0_initialized = True

        ekf.processImu(imu)

        if ts >= gts[pose_corr_idx][0]:
            pose_corr = gt_pose0 @ gts[pose_corr_idx][1]

            ekf.processPose(pose_corr)

            gt_poses.append(pose_corr)
            gt_t.append(ekf.ts)
            res_poses.append(ekf.nav.pose_mat())
            if pose_corr_idx + 1 < len(gts):
                pose_corr_idx += 1

        if dur_t > 0 and ts - first_ts - start_ts > dur_t:
            break

    print(f"scanned duration: {ts - first_ts - start_ts:0.04} s")
    print(f"updates num: {len(res_poses)}\n")
    if res_poses:
        ate_rot, ate_trans = calc_ate(res_poses, gt_poses)
        print(f"ATE_rot:   {ate_rot:.04f} deg")
        print(f"ATE trans: {ate_trans:.04f} m")

    if not ekf._logging or not ekf._navs:
        return

    if plot == "graphs":
        ekf_graphs(ekf,
                   xy_plot=xy_plot,
                   gt=(gt_t, gt_poses),
                   labels=["ES EKF IMU + GT pose correction", "GT poses"])
    elif plot == "point_viz":
        ekf_viz(ekf)
    elif not plot:
        return
    else:
        print(f"WARNING: plot param '{plot}' doesn't "
              f"supported")


@click.command(name="ouster")
@click.argument("file", required=True, type=click.Path(exists=True))
@click.option(
    "-m",
    "--meta",
    required=False,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help=
    "Metadata for PCAP/BAG, required if automatic metadata resolution fails")
@click.option("--start-scan", type=int, default=0, help="Start scan number")
@click.option("--end-scan", type=int, help="End scan number, inclusive")
@click.option("-p",
              "--plot",
              required=False,
              type=str,
              help="Plotting option [graphs, point_viz]")
@click.option('--use-imu-prediction',
              is_flag=True,
              help="Use EKF IMU pose prediction for KissICP register frame, "
              "i.e. lously coupled Lidar Inertial Odometry")
@click.option('--use-gt-guess',
              is_flag=True,
              help="Use GT pose as a guess for KissICP (used solely for "
              "sanity testing)")
@click.option("-g",
              "--gt-file",
              required=False,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="Ground truth file with poses to compare "
              "(Newer College format)")
@click.option("--kiss-min-range",
              type=float,
              default=1,
              help="KissICP min range param in m (default 1)")
@click.option("--kiss-max-range",
              type=float,
              default=70,
              help="KissICP max range param in m (default 70)")
@click.option("--beams",
              type=int,
              default=0,
              help="Active beams number in a lidar scan (i.e. reduces "
              "beams (i.e. active rows of a scan) to the NUM)")
@click.option(
    "--save-kitti-poses",
    required=False,
    type=click.Path(exists=False, dir_okay=False, readable=True),
    help=
    "Save resulting poses to the file (in kitti format)")
@click.option(
    "--save-nc-gt-poses",
    required=False,
    type=click.Path(exists=False, dir_okay=False, readable=True),
    help=
    "Save resulting poses to the file (in NC ground truth format)")
def ptudes_ekf_ouster(file: str,
                      meta: Optional[str],
                      start_scan: int,
                      end_scan: Optional[int] = None,
                      plot: Optional[str] = None,
                      use_imu_prediction: bool = False,
                      use_gt_guess: bool = False,
                      gt_file: Optional[str] = None,
                      beams: int = 0,
                      save_kitti_poses: Optional[str] = None,
                      save_nc_gt_poses: Optional[str] = None,
                      kiss_min_range: float = 1.0,
                      kiss_max_range: float = 70.0) -> None:
    """EKF with Ouster IMUs PCAP/BAG and scan KissICP poses updates.

    Essentially a smoothing action to the KissICP trajectory output,
    which is not even a lousely coupled IMU, because KissICP
    state is not changed in any way (i.e. map, prediction model,
    deskew, etc not affected)

    Ground truth (--gt-file) in Newer College dataset is used for graphs (--plot
    graphs) to compare the resulting trajectory or for sanity testing EKF with
    --use-gt-guess when GT used for pose corrections.

    Use save trajectory in kitti format (--save-kitt-poses) or NC GT format
    (--save-nc-gt-poses) to visualize 3d point clouds of the map using `ptudes
    flyby` or other tools.

    Compare trajectories of different runs using the (--save-nc-gt-poses) and
    `ptudes ekf-bench cmp` command.

    Use --beams NUM to reduce the number of beams of lidar scan, which is useful
    to simulate low res sensors.
    """

    if not gt_file and use_gt_guess:
        print("ERROR: --use-gt-guess requires the GT poses (--gt-file)")
        exit(1)

    meta = resolve_metadata(file, meta)
    if not meta:
        raise click.ClickException(
            "File not found, please specify a metadata file with `-m`")
    info = read_metadata_json(meta)

    log_metrics = bool(plot)

    display_header = f"data path: {file}\n"
    display_header += f"metadata path: {meta}\n\n"
    display_header += f"scans range: {start_scan} - {end_scan}\n"
    display_header += f"kiss min/max: {kiss_min_range} - {kiss_max_range}\n"
    display_header += f"use-imu-prediction: {use_imu_prediction}, use-gt-guess: {use_gt_guess}\n"
    display_header += f"beams: {beams or info.format.pixels_per_column}\n"
    display_header += f"sensor: {info.prod_line}, {info.mode}\n"
    print(display_header)
    print(f"metrics logging: {log_metrics}")

    packet_source = read_packet_source(file, meta=info)

    imu_to_sensor = packet_source.metadata.imu_to_sensor_transform.copy()
    imu_to_sensor[:3, 3] /= 1000  # mm to m convertion
    sensor_to_imu = np.linalg.inv(imu_to_sensor)

    # exploiting extrinsics mechanics of Ouster SDK to
    # make an XYZLut that transforms lidar points to the
    # imu frame (nav frame in our case)
    packet_source.metadata.extrinsic = sensor_to_imu

    data_source = OusterLidarData(packet_source)

    kiss_icp = KissICPWrapper(packet_source.metadata,
                              _use_extrinsics=True,
                              _min_range=kiss_min_range,
                              _max_range=kiss_max_range)

    stats = StreamStatsTracker(use_beams_num=32, metadata=data_source.metadata)


    ekf = ESEKF(_logging=log_metrics)

    res_t = []
    kiss_poses = []

    res_poses = []

    t_imu = 0
    t_imu_cnt = 0

    t_corr = 0
    t_corr_cnt = 0

    t_kiss = 0

    scans_total = end_scan - start_scan if end_scan else None
    scan_tqdm_it = iter(tqdm(cycle([0]), total=scans_total, unit=" scan"))

    t_track = 0

    init_stats_ts = 3
    init_stats_flag = True

    gts = []
    gt_traj = None
    gt_traj_first = False
    gt_traj0 = np.eye(4)
    if gt_file:
        gts = read_newer_college_gt(gt_file)
        if use_gt_guess:
            gt_traj = pu.TrajectoryEvaluator(gts, time_bounds=1.0)

    imus_per_scan = 1

    for scan_idx, d in data_source.withScanIdx(start_scan=start_scan,
                                               end_scan=end_scan):

        if isinstance(d, IMU):
            t1 = time.monotonic()
            stats.trackImu(d)
            t_track += time.monotonic() - t1

            t1 = time.monotonic()
            ekf.processImu(d)
            t_imu += time.monotonic() - t1
            t_imu_cnt += 1

            imus_per_scan += 1

        elif isinstance(d, client.LidarScan):
            # yep, just count and time the scan iteration
            next(scan_tqdm_it)

            if not imus_per_scan:
                # skipping scans that doesn't have IMUs in between, this may
                # happen when a stray lidar packet appears and breaks the stream
                # of scans in a ScanBatcher (i.e. scans composed of packets of
                # different frames)
                continue
            imus_per_scan = 0

            ls = d

            t1 = time.monotonic()
            stats.trackScan(ls)
            t_track += time.monotonic() - t1

            if beams:
                reduce_active_beams(ls, beams)

            t1 = time.monotonic()

            ts = client.last_valid_column_ts(ls) * 1e-09

            if use_imu_prediction:
                # EKF IMU based prediction for KissICP
                pose_guess = ekf.nav.pose_mat()
            elif use_gt_guess and gt_traj is not None:
                # GT pose prediction for KissICP (for sanity tests)
                gt_guess = gt_traj.pose_at(ts)
                if not gt_traj_first:
                    gt_traj0 = np.linalg.inv(gt_guess)
                    gt_traj_first = True
                pose_guess = gt_traj0 @ gt_guess
            else:
                # Standard constant velocity/linear KissICP prediction
                prediction = kiss_icp._kiss.get_prediction_model()
                last_pose = (kiss_icp._kiss.poses[-1]
                             if kiss_icp._kiss.poses else np.eye(4))
                pose_guess = last_pose @ prediction

            t1 = time.monotonic()
            kiss_icp.register_frame(ls, initial_guess=pose_guess)
            t_kiss += time.monotonic() - t1

            t1 = time.monotonic()
            ekf.processPose(kiss_icp.pose)
            t_corr += time.monotonic() - t1
            t_corr_cnt += 1

            # collect metrics
            kiss_poses.append(kiss_icp.pose)
            res_poses.append(ekf.nav.pose_mat())

            res_t.append(ekf.ts)

        if not init_stats_flag and stats.dt > init_stats_ts:
            print(stats)
            grav_est = stats.acc_mean / np.linalg.norm(stats.acc_mean)
            print("Grav vector est: ", grav_est)
            init_stats_flag = True



    # log header for storing resulting poses
    header = display_header
    header += f"(scans/updates num: {len(res_poses)})\n"
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    header += f"time: {now}"

    if save_kitti_poses:
        save_poses_kitti_format(save_kitti_poses, res_poses, header=header)
        print(f"Kitti poses saved to: {save_kitti_poses}")

    if save_nc_gt_poses:
        save_poses_nc_gt_format(save_nc_gt_poses,
                                t=res_t,
                                poses=res_poses,
                                header=header)
        print(f"NC GT poses saved to: {save_nc_gt_poses}")

    if t_imu_cnt and t_corr_cnt:
        print("\nTimings:")
        print(f"  ESEKF imu process:      {t_imu / t_imu_cnt:.05f} s per step")
        print(f"  ESEKF update:           {t_corr / t_corr_cnt:.05f} s per update")
        print(f"  KissICP register frame: {t_kiss / t_corr_cnt:.05f} s per frame")
        print(f"  Stats tracking:         {t_track / t_corr_cnt:.05f} s per frame")



    if plot == "graphs":
        gt2 = None
        if gts:
            # aligning/filtering to have only close by ts gt and calc poses
            gts, res_t_matched = filter_nc_gt_by_close_ts(gts, res_t)
            kiss_poses_matched = []
            res_poses_matched = []
            idx = 0
            for t_m in res_t_matched:
                while res_t[idx] != t_m:
                    idx += 1
                kiss_poses_matched.append(kiss_poses[idx])
                res_poses_matched.append(res_poses[idx])
                idx += 1

            if gts:
                # GT file has poses for the processed timestampss
                gts_pose0 = res_poses_matched[0] @ np.linalg.inv(gts[0][1])
                gt2_t = [g[0] for g in gts]
                gt2_poses = [gts_pose0 @ g[1] for g in gts]
                gt2 = (gt2_t, gt2_poses)

                num_poses = len(gt2_poses)

                ate_rot, ate_trans = calc_ate(res_poses_matched, gt2_poses)
                print(f"\nGround truth comparison (with ES EKF smoothing "
                      f"{num_poses} poses):")
                print(f"ATE_rot:   {ate_rot:.04f} deg")
                print(f"ATE trans: {ate_trans:.04f} m")

                ate_rot, ate_trans = calc_ate(kiss_poses_matched, gt2_poses)
                print("\nGround truth comparison (no-EKF, only KissICP "
                      f"{num_poses} poses):")
                print(f"ATE_rot:   {ate_rot:.04f} deg")
                print(f"ATE trans: {ate_trans:.04f} m")

                # graph only matched gt poses of kiss icp too (if gt-file
                # is present)
                res_t = res_t_matched
                kiss_poses = kiss_poses_matched

        if ekf._logging and ekf._navs:
            ekf_graphs(ekf,
                       gt=(res_t, kiss_poses),
                       gt2=gt2,
                       xy_plot=True,
                       labels=[
                           "ES EKF KissICP smoothed poses",
                           "KissICP only poses", "GT poses"
                       ])

        # TODO: extract me ... ?
        import matplotlib.pyplot as plt
        rel_t = np.array(kiss_icp._poses_ts) - kiss_icp._poses_ts[0]
        plt.plot(rel_t, kiss_icp._err_dt, label="KissICP: trans error (m)")
        plt.plot(rel_t, kiss_icp._err_drot, label="KissICP: rotation error (rad)")
        plt.plot(rel_t, kiss_icp._sigmas, label="KissICP: adaptive threshold sigma")
        plt.grid(True)
        plt.xlabel("t (s)")
        plt.legend(loc="upper right")
        plt.show()

    elif plot == "point_viz":
        ekf_viz(ekf)
    elif not plot:
        return
    else:
        print(f"WARNING: plot param '{plot}' doesn't supported")


@click.command(name="cmp")
@click.argument("gt_file", required=True, type=click.Path(exists=True))
@click.argument("gt_file_cmp",
                required=False,
                type=click.Path(exists=True),
                nargs=-1)
@click.option("-p",
              "--plot",
              required=False,
              type=str,
              help="Plotting option [graphs, graphs_full, point_viz]")
@click.option('--use-gt-frame',
              is_flag=True,
              help="Use GT frame (i.e. align cmp poses to gt)")
@click.option('--xy-plot',
              is_flag=True,
              help="Draw X and Y dimenstions on XY plane, instead of separate")
def ptudes_ekf_cmp(gt_file: str,
                   gt_file_cmp: List[str],
                   plot: Optional[str] = None,
                   xy_plot: bool = False,
                   use_gt_frame: bool = False) -> None:
    """Compare trajectories in Newer College Dataset Formats"""

    gts_all = read_newer_college_gt(gt_file)

    gts_cmp_all = [read_newer_college_gt(f) for f in gt_file_cmp]

    gts = []
    gts_cmp = []

    for gc in gts_cmp_all:
        gts_el, gts_cmp_el = filter_nc_gt_by_cmp(gts_all, gc)
        gts.append(gts_el)
        gts_cmp.append(gts_cmp_el)

    fname = lambda f: os.path.splitext(os.path.basename(f))[0]

    for idx, cmp_file in enumerate(gt_file_cmp):
        gts_poses = [p for (_, p) in gts[idx]]
        gts_cmp_poses = [p for (_, p) in gts_cmp[idx]]
        ate_rot, ate_trans = calc_ate(gts_poses, gts_cmp_poses)
        print(f"\nTraj poses comparisons GT v. {fname(cmp_file)} "
              f"({len(gts_poses)} poses):")
        print(f"ATE_rot:   {ate_rot:.04f} deg")
        print(f"ATE trans: {ate_trans:.04f} m")

    if plot in ["graphs", "graphs_full"]:

        if len(gt_file_cmp) != 1:
            use_gt_frame = True
            print("\nNOTE: Enforcing --use-gt-frame because the number of "
                  "trajectories to compare is zero or more than one")

        if not gts_cmp and plot == "graphs":
            plot = "graphs_full"


        # combined GT for all trajectories to compare timestamps
        gts_comb_cmp = []
        if gts_cmp:
            cmp_min_ts = min([gc[0][0] for gc in gts_cmp])
            cmp_max_ts = max([gc[-1][0] for gc in gts_cmp])
            gts_comb_cmp = [
                gc for gc in gts_all if gc[0] >= cmp_min_ts and gc[0] <= cmp_max_ts
            ]

        if not use_gt_frame:
            # only one gt_file_cmp is present
            gts_pose0 = gts_cmp[0][0][1] @ np.linalg.inv(gts_comb_cmp[0][1])
            gts_comb_cmp = [(t, gts_pose0 @ p) for t, p in gts_comb_cmp]
            gts_all = [(t, gts_pose0 @ p) for t, p in gts_all]
        else:
            # move cmp poses to the gt poses (i.e. align)
            for idx in range(len(gts_cmp)):
                gts_cmp_pose0 = gts[idx][0][1] @ np.linalg.inv(
                    gts_cmp[idx][0][1])
                gts_cmp[idx] = [(t, gts_cmp_pose0 @ p)
                                for t, p in gts_cmp[idx]]

        cmp_labels = [
            f"Cmp poses {i + 1}: {fname(f)}"
            for i, f in enumerate(gt_file_cmp)
        ]

        gt_poses_graphs(
            [gts_all if plot == "graphs_full" else gts_comb_cmp, *gts_cmp],
            xy_plot=xy_plot,
            labels=[f"GT Poses: {fname(gt_file)}", *cmp_labels])
    elif plot == "point_viz":
        print("PointViz view of the trajectories to compare NOT YET "
              "IMPLEMENTED")


ptudes_ekf_bench.add_command(ptudes_ekf_sim)
ptudes_ekf_bench.add_command(ptudes_ekf_nc)
ptudes_ekf_bench.add_command(ptudes_ekf_ouster)
ptudes_ekf_bench.add_command(ptudes_ekf_cmp)
