"""Error-state Extended Kalman Filter experiments"""
from typing import Iterator
import time
import click
from typing import Optional

import ouster.client as client
from ouster.sdk.util import resolve_metadata

from ptudes.utils import (read_metadata_json, read_packet_source,
                          read_newer_college_gt, save_poses_kitti_format,
                          filter_nc_gt_by_ts, filter_nc_gt_by_close_ts,
                          reduce_active_beams)
from ptudes.lio_ekf import LioEkfScans

import numpy as np
import numpy.random as npr
from ptudes.ins.data import (IMU, GRAV, calc_ate, ekf_traj_ate,
                             _collect_navs_from_gt, StreamStatsTracker)
from ptudes.ins.es_ekf import ESEKF
from ptudes.ins.viz_utils import (lio_ekf_viz, lio_ekf_graphs,
                                  lio_ekf_error_graphs)
from ptudes.data import OusterLidarData, last_valid_packet_ts
from ptudes.kiss import KissICPWrapper
from tqdm import tqdm
from itertools import cycle

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

    Ground truth pose is an integrated noise/bias free measurements.
    """

    print("Using sim IMUs with params:")
    print(f"  freq: {freq} Hz")
    print(f"  acc_noise_std: {acc_noise_std}")
    print(f"  gyr_noise_std: {gyr_noise_std}")
    print(f"  correction dt: {corr_t:.02} s")

    print("Running EKF ... \n")

    ekf_gt = ESEKF()
    ekf = ESEKF()

    initialized = False
    dur_t = duration
    ts = 0
    start_ts = 0
    last_corr_t = 0
    # gt_t = []
    # gt_poses = []
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
            # gt_t.append(ts)
            # gt_poses.append(ekf_gt._nav_curr.pose_mat())

        if ts - start_ts > dur_t:
            break

    print("Results:")

    print(f"processed duration: {ts - start_ts:0.04} s")
    print(f"updates num: {len(ekf._nav_scan_idxs)}\n")

    print("NAV GT:\n", ekf_gt._nav_curr)
    print("NAV:\n", ekf._nav_curr)

    ate_rot, ate_trans = ekf_traj_ate(ekf_gt, ekf)
    print(f"ATE_rot:   {ate_rot:.04f} deg")
    print(f"ATE trans: {ate_trans:.04f} m")

    # get associated nav states with gt states and time
    gt_t, gt_navs, navs = _collect_navs_from_gt(ekf_gt, ekf)
    gt_poses = [nav.pose_mat() for nav in gt_navs]
    print("gt_poses.shape = ", len(gt_poses))

    if plot == "graphs":
        lio_ekf_graphs(ekf, gt=(gt_t, gt_poses))
        # lio_ekf_graphs(ekf_gt, gt=(gt_t, gt_poses))
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
@click.option("-i",
              "--imu-topic",
              required=False,
              default="/alphasense_driver_ros/imu",
              type=str,
              help="Imu topic name to use (msg/Imu or imu_packets)")
def ptudes_ekf_nc(file: str,
                  meta: Optional[str] = None,
                  gt_file: Optional[str] = None,
                  duration: float = 2.0,
                  start_ts: float = 0.0,
                  plot: Optional[str] = None,
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

    ekf = ESEKF(init_grav=init_grav)

    gt_t = []
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
                pose_corr_idx += 1
                gt_pose0 = np.linalg.inv(gts[pose_corr_idx][1])
            continue

        ekf.processImu(imu)

        if ts >= gts[pose_corr_idx][0]:
            pose_corr = gt_pose0 @ gts[pose_corr_idx][1]

            ekf.processPose(pose_corr)

            gt_poses.append(pose_corr)
            gt_t.append(imu.ts)  # TODO: fix me on switch to continuous time!
            if pose_corr_idx + 1 < len(gts):
                pose_corr_idx += 1

        if dur_t > 0 and ts - first_ts - start_ts > dur_t:
            break

    nav_poses = [ekf._navs[i].pose_mat() for i in ekf._nav_scan_idxs]

    print(f"scanned duration: {ts - first_ts - start_ts:0.04} s")
    print(f"updates num: {len(nav_poses)}\n")
    if nav_poses:
        ate_rot, ate_trans = calc_ate(nav_poses, gt_poses)
        print(f"ATE_rot:   {ate_rot:.04f} deg")
        print(f"ATE trans: {ate_trans:.04f} m")

    if not ekf._navs:
        return

    if plot == "graphs":
        lio_ekf_graphs(ekf, gt=(gt_t, gt_poses))
        # lio_ekf_graphs(ekf_gt)
        # lio_ekf_error_graphs(ekf_gt, ekf)
    elif plot == "point_viz":
        lio_ekf_viz(ekf)
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
@click.option("-g",
              "--gt-file",
              required=False,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="Ground truth file with poses to compare "
              "(Newer College format)")
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
def ptudes_ekf_ouster(file: str,
                      meta: Optional[str],
                      start_scan: int,
                      end_scan: Optional[int] = None,
                      plot: Optional[str] = None,
                      use_imu_prediction: bool = False,
                      gt_file: Optional[str] = None,
                      beams: int = 0,
                      save_kitti_poses: Optional[str] = None,
                      kiss_max_range: float = 70.0) -> None:
    """EKF with Ouster IMUs PCAP/BAG and scan KissICP poses updates.

    Essentially a smoothing action to the KissICP trajectory output,
    which is not even a lousely coupled IMU, because KissICP
    state is not changed in any way (i.e. map, prediction model,
    deskew, etc not affected)

    Ground truth (--gt-file) in Newer College dataset is used only
    for graphs (--plot graphs) to compare the resulting trajectory.

    Use save trajectory in kitti format (--save-kitt-poses) to visualize
    3d point clouds of the map using `ptudes flyby` or other tools.
    """

    meta = resolve_metadata(file, meta)
    if not meta:
        raise click.ClickException(
            "File not found, please specify a metadata file with `-m`")
    click.echo(f"Reading metadata from: {meta}")
    info = read_metadata_json(meta)

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
                              _min_range=1.0,
                              _max_range=kiss_max_range)

    stats = StreamStatsTracker(use_beams_num=32, metadata=data_source.metadata)

    ekf = ESEKF()

    scan_idx = 0

    gt_t = []
    gt_poses = []

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
    init_stats_flag = False

    for d in data_source:

        if isinstance(d, IMU):
            if scan_idx >= start_scan:
                t1 = time.monotonic()
                stats.trackImu(d)
                t_track += time.monotonic() - t1

                t1 = time.monotonic()
                ekf.processImu(d)
                t_imu += time.monotonic() - t1
                t_imu_cnt += 1

        elif isinstance(d, client.LidarScan):
            if scan_idx < start_scan:
                scan_idx += 1
                continue

            # yep, just count and time the scan iteration
            next(scan_tqdm_it)

            ls = d

            t1 = time.monotonic()
            stats.trackScan(ls)
            t_track += time.monotonic() - t1

            if beams:
                reduce_active_beams(ls, beams)

            t1 = time.monotonic()

            if use_imu_prediction:
                # EKF IMU based prediction for KissICP
                pose_guess = ekf._nav_curr.pose_mat()
            else:
                # Standard constant velocity/linear KissICP prediction
                prediction = kiss_icp._kiss.get_prediction_model()
                last_pose = (kiss_icp._kiss.poses[-1]
                             if kiss_icp._kiss.poses else np.eye(4))
                pose_guess = last_pose @ prediction
            kiss_icp.register_frame(ls, initial_guess=pose_guess)
            t_kiss += time.monotonic() - t1

            t1 = time.monotonic()
            ekf.processPose(kiss_icp.pose)


            t_corr += time.monotonic() - t1
            t_corr_cnt += 1

            # print(f"\n\nimu iter[{scan_idx}] = ", t_imu / t_imu_cnt)
            # print(f"corr iter[{scan_idx}] = ", t_corr / t_corr_cnt)
            # print(f"kiss iter[{scan_idx}] = ", t_kiss / t_corr_cnt)
            # print(f"track iter[{scan_idx}] = ", t_track / t_corr_cnt)

            res_poses.append(ekf._nav_curr.pose_mat())

            gt_poses.append(kiss_icp.pose)
            gt_t.append(ekf._navs_t[-1])

            scan_idx += 1

            if end_scan is not None and scan_idx > end_scan:
                break

        if not init_stats_flag and stats.dt > init_stats_ts:
            print(stats)
            grav_est = stats.acc_mean / np.linalg.norm(stats.acc_mean)
            print("Grav vector est: ", grav_est)
            init_stats_flag = True

    if not ekf._navs:
        return

    if save_kitti_poses:
        save_poses_kitti_format(save_kitti_poses, res_poses)
        print(f"Kitti poses saved to: {save_kitti_poses}")


    if plot == "graphs":
        gt2 = None
        if gt_file:
            gts = read_newer_college_gt(gt_file)

            # aligning/filtering to have only close by ts gt and calc poses
            gts, gt_t_matched = filter_nc_gt_by_close_ts(gts, gt_t)
            gt_poses_matched = []
            res_poses_matched = []
            idx = 0
            for t_m in gt_t_matched:
                while gt_t[idx] != t_m:
                    idx += 1
                gt_poses_matched.append(gt_poses[idx])
                res_poses_matched.append(res_poses[idx])
                idx += 1

            if gts:
                gts_pose0 = np.linalg.inv(gts[0][1])
                gt2_t = [g[0] for g in gts]
                gt2_poses = [gts_pose0 @ g[1] for g in gts]
                gt2 = (gt2_t, gt2_poses)

                num_poses = len(gt2_poses)

                ate_rot, ate_trans = calc_ate(res_poses_matched, gt2_poses)
                print(f"\nGround truth comparison (with smoothed EKF "
                      f"{num_poses} poses):")
                print(f"ATE_rot:   {ate_rot:.04f} deg")
                print(f"ATE trans: {ate_trans:.04f} m")

                ate_rot, ate_trans = calc_ate(gt_poses_matched, gt2_poses)
                print("\nGround truth comparison (no-EKF, only KissICP "
                      f"{num_poses} poses):")
                print(f"ATE_rot:   {ate_rot:.04f} deg")
                print(f"ATE trans: {ate_trans:.04f} m")

                # graph only matched gt poses of kiss icp too (if gt-file
                # is present)
                gt_t = gt_t_matched
                gt_poses = gt_poses_matched

                # ate_rot, ate_trans = calc_ate(gt_poses, gt2_poses)
                # print(f"ATE_rot:   {ate_rot:.04f} deg")
                # print(f"ATE trans: {ate_trans:.04f} m")

                # dts = [t2-t1 for t1, t2 in zip(gt_t_matched, gt2_t)]
                # print("dts = ", dts)
        lio_ekf_graphs(
            ekf,
            gt=(gt_t, gt_poses),
            gt2=gt2,
            labels=["ES EKF smoothed poses", "KissICP poses", "GT poses"])
    elif plot == "point_viz":
        lio_ekf_viz(ekf)
    elif not plot:
        return
    else:
        print(f"WARNING: plot param '{plot}' doesn't supported")


ptudes_ekf_bench.add_command(ptudes_ekf_sim)
ptudes_ekf_bench.add_command(ptudes_ekf_nc)
ptudes_ekf_bench.add_command(ptudes_ekf_ouster)
