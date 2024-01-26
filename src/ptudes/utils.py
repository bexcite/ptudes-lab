from typing import Optional, Callable, List, Tuple

import os
import glob
import json
import numpy as np
from pathlib import Path

import weakref
import ouster.client as client
import ouster.viz as viz
import ouster.pcap as pcap
from ouster.viz import (PointViz, ScansAccumulator, add_default_controls)
import ouster.sdk.pose_util as pu

from scipy.spatial.transform import Rotation

from ptudes.bag import OusterRawBagSource

# NC 2021 apply transform to move GT from base to Ouster IMU Nav frame
# transforms from newer_college_2021/os_imu_lidar_transforms.yaml
NC_OS_IMU_TO_OS_SENSOR = np.eye(4)
NC_OS_IMU_TO_OS_SENSOR[:3, 3] = [-0.014, 0.012, 0.015]
NC_OS_SENSOR_TO_BASE = np.eye(4)
NC_OS_SENSOR_TO_BASE[:3, 3] = [0.001, 0.000, 0.091]
NC_OS_IMU_TO_BASE = NC_OS_SENSOR_TO_BASE @ NC_OS_IMU_TO_OS_SENSOR

def vee(vec: np.ndarray) -> np.ndarray:
    w = np.zeros((3, 3))
    w[0, 1] = -vec[2]
    w[0, 2] = vec[1]
    w[1, 0] = vec[2]
    w[1, 2] = -vec[0]
    w[2, 0] = -vec[1]
    w[2, 1] = vec[0]
    return w


def spin(pviz: PointViz,
         on_update: Callable[[PointViz, float], None],
         *,
         period: float = 0.0333,
         total: float = 0) -> None:
    import threading
    import time

    quit = threading.Event()

    def animate() -> None:
        first_tick_ts = 0.0
        last_tick_ts = 0.0
        while not quit.is_set():
            dt = time.monotonic() - last_tick_ts
            time.sleep(period - dt if period - dt > 0 else 0)
            last_tick_ts = time.monotonic()
            if not first_tick_ts:
                first_tick_ts = last_tick_ts
            on_update(pviz, last_tick_ts - first_tick_ts)
            pviz.update()

            if total > 0 and time.monotonic() - first_tick_ts > total:
                break

    thread = threading.Thread(target=animate)
    thread.start()

    pviz.run()
    quit.set()
    thread.join()


def make_point_viz(title: str = "", show_origin: bool = True) -> PointViz:
    point_viz = PointViz(f"Ptudes Viz {title}")

    weakself = weakref.ref(point_viz)

    def handle_keys(ctx, key, mods) -> bool:
        self = weakself()
        if self is None:
            return True
        if key == 256:  # ESC
            self.running(False)
        return True

    point_viz.push_key_handler(handle_keys)
    add_default_controls(point_viz)

    point_viz.camera.set_yaw(140)
    point_viz.camera.set_pitch(0)
    point_viz.camera.set_dolly(-100)
    point_viz.camera.set_fov(90)
    point_viz.target_display.set_ring_size(3)
    point_viz.target_display.enable_rings(True)

    if show_origin:
        viz.AxisWithLabel(point_viz,
                          pose=np.eye(4),
                          label="O",
                          thickness=5,
                          length=1,
                          label_scale=1,
                          enabled=True)

    return point_viz


def estimate_apex_dolly(min_max: np.ndarray, fov_deg: float) -> float:
    """Estimate the dolly so it fit majority of points between min/max"""
    d = np.linalg.norm(min_max[:, 1] - min_max[:, 0])
    D = 1.4142 * d / np.sin(fov_deg * np.pi / 180)
    return max(-100, 100 * np.log(max(0.001, D) / 50.0))


def map_points_num(sa: ScansAccumulator) -> int:
    """Helper to extract the number of points in the map"""
    if sa._map_overflow:
        return sa._map_xyz.shape[0]
    else:
        return sa._map_idx


def prune_trajectory(traj_poses: pu.TrajPoses,
                     min_dist_m: Optional[float] = 5,
                     min_dist_angle: Optional[float] = 5,
                     start_idx: Optional[int] = None,
                     end_idx: Optional[int] = None) -> pu.TrajPoses:
    """Make pruned poses by removing close to each other knots

    TODO: Stretch goal - generate a B-Spline for smoother camera moves :)
    """
    start_idx = 0 if start_idx is None else start_idx
    end_idx = len(traj_poses) - 1 if end_idx is None else end_idx
    assert start_idx <= end_idx, \
        f"{start_idx = } should be lte {end_idx = }"
    assert start_idx < len(traj_poses) and end_idx < len(traj_poses), \
        f"{start_idx = } and {end_idx = } should be lt {len(traj_poses) = }"

    pruned_poses = [traj_poses[start_idx]]
    last_pose_inv = np.linalg.inv(pruned_poses[0][1])
    idx = start_idx + 1
    for tp in traj_poses[idx:end_idx+1]:
        p = tp[1]
        pd = pu.log_pose(last_pose_inv @ p)
        pda = np.linalg.norm(pd[:3])
        pdm = np.linalg.norm(pd[3:])
        if pda > min_dist_angle * np.pi / 180 or pdm > min_dist_m or idx == end_idx:
            pruned_poses.append(tp)
            last_pose_inv = np.linalg.inv(p)
        idx += 1
    # HACK: handle edge case when just one scan is selected and add the next
    # one for traj evaluator to work ...
    if len(pruned_poses) < 2 and end_idx + 1 < len(traj_poses):
        pruned_poses.append(traj_poses[end_idx + 1])
    return pu.make_kiss_traj_poses([p for _, p in pruned_poses])


def read_metadata_json(meta_path: str) -> Optional[client.SensorInfo]:
    with open(meta_path) as json_file:
        json_str = json_file.read()
        js = json.loads(json_str)
        # HACK: backfill lidar mode to make newer college dataset 2020
        # beam_extrinsics parseable by ouster-sdk :()
        if ("beam_altitude_angles" in js and "beam_azimuth_angles" in js
                and "lidar_mode" not in js):
            print("WARNING: lidar_mode is not present in legacy metadata "
                  f"'{meta_path}' so using lidar_mode: 1024x10")
            js["lidar_mode"] = "1024x10"
        return client.SensorInfo(json.dumps(js))


def read_packet_source(
        file_path: str,
        meta: Optional[client.SensorInfo] = None) -> client.PacketSource:
    """Open PCAP of BAG based Ouster raw packet source."""

    file = Path(file_path)
    if file.is_file():
        if file.suffix == ".pcap":
            return pcap.Pcap(file_path, meta)
        elif file.suffix == ".bag":
            return OusterRawBagSource(file, meta)
    elif file.is_dir():
        # TODO: natural sort? for newer college dataset is not needed
        # so maybe some time later
        bags_paths = sorted(
            [Path(p) for p in glob.glob(str(Path(file) / "*.bag"))])
        return OusterRawBagSource(bags_paths, meta)


# heavily inspired by KissICP method
def save_poses_kitti_format(filename: str,
                            poses: List[np.ndarray],
                            header: str = ""):
    kitti_format_poses = np.array(
        [np.concatenate((pose[0], pose[1], pose[2])) for pose in poses])
    np.savetxt(fname=filename, X=kitti_format_poses, header=header)


def save_poses_nc_gt_format(filename: str, t: List[float],
                            poses: List[np.ndarray],
                            header: str = ""):
    """Save odom result in the Newer College GT format for comparison."""
    t_arr = np.array(t)
    poses_arr = np.array(poses)

    # saving in the BASE frame, assuming that incoming poses in IMU (nav frame)
    # but for uesrs who care about save/recover functionality
    # these transforms on save/restore essentially invariant (unless
    # other read functions are used in other places with other assumptions)
    os_base_to_imu = np.linalg.inv(NC_OS_IMU_TO_BASE)
    poses_arr = np.einsum("nij,jk->nik", poses_arr, os_base_to_imu)

    res = np.zeros((len(t), 9))
    # secs
    res[:, 0] = np.floor(t_arr)
    # nsecs
    res[:, 1] = np.floor((t_arr - res[:, 0]) * 1e+9)
    # x,y,z
    res[:, 2:5] = poses_arr[:, :3, 3]
    # qx,qy,qz,qw
    res[:, 5:10] = Rotation.from_matrix(poses_arr[:, :3, :3]).as_quat()


    data_spec = "sec,nsec,x,y,z,qx,qy,qz,qw"
    if header:
        header += "\n\n" + data_spec

    np.savetxt(fname=filename, X=res, delimiter=", ", header=header)


def read_newer_college_gt(
        data_path: str,
        to_os_imu: bool = True) -> List[Tuple[float, np.ndarray]]:
    """Read ground truth poses for Newer College 2021 dataset.
    
    Return:
        List of tuples (ts, pose4x4), where pose4x4 converted to the Ouster
        IMU NavFrame.
    """
    gt_data = np.loadtxt(data_path, delimiter=",")
    ts = gt_data[:, 0] + gt_data[:, 1] * 1e-9

    pos = np.tile(np.eye(4), reps=(gt_data.shape[0], 1, 1))
    pos[:, :3, 3] = gt_data[:, 2:5]

    rots = Rotation.from_quat(gt_data[:, 5:9]).as_matrix()
    pos[:, :3, :3] = rots

    if to_os_imu:
        pos = np.einsum("nij,jk->nik", pos, NC_OS_IMU_TO_BASE)

    return [(t, p) for t, p in zip(ts[:], pos[:])]


def filter_nc_gt_by_close_ts(
        nc_gt,
        gt_t) -> Tuple[List[Tuple[float, np.ndarray]], List[float]]:
    if not len(nc_gt):
        return nc_gt
    if not len(gt_t):
        return []

    # assuming non-decreasing timestamps
    nc_t = [g[0] for g in nc_gt]
    min_nc_t = np.min(np.array(nc_t[1:]) - np.array(nc_t[:-1]))
    min_gt_t = np.min(np.array(gt_t[1:]) - np.array(gt_t[:-1]))
    min_dt = min(min_nc_t, min_gt_t)

    res_nc_gt = []
    res_gt_t = []

    nc_gt_it = iter(nc_gt)
    gt_t_it = iter(gt_t)

    n_t = next(nc_gt_it)
    g_t = next(gt_t_it)

    while True:
        try:
            while abs(n_t[0] - g_t) > min_dt:
                while n_t[0] < g_t - min_dt:
                    n_t = next(nc_gt_it)
                while g_t < n_t[0] - min_dt:
                    g_t = next(gt_t_it)
            if n_t[0] < g_t:
                n_t2 = next(nc_gt_it)
                if abs(n_t[0] - g_t) < abs(n_t2[0] - g_t):
                    res_nc_gt.append(n_t)
                    res_gt_t.append(g_t)
                    n_t = n_t2
                    g_t = next(gt_t_it)
            elif g_t <= n_t[0]:
                g_t2 = next(gt_t_it)
                if abs(n_t[0] - g_t) < abs(n_t[0] - g_t2):
                    res_nc_gt.append(n_t)
                    res_gt_t.append(g_t)
                    n_t = next(nc_gt_it)
                    g_t = g_t2
        except StopIteration:
            break

    return res_nc_gt, res_gt_t


def filter_nc_gt_by_cmp(
    nc_gt, nc_gt_cmp
) -> Tuple[List[Tuple[float, np.ndarray]], List[Tuple[float, np.ndarray]]]:
    """Find the closest subset of nc_gt_cmp in nc_gt poses."""

    gt_cmp_t = [g[0] for g in nc_gt_cmp]

    gt_matched, gt_cmp_t_matched = filter_nc_gt_by_close_ts(nc_gt, gt_cmp_t)
    gt_cmp_poses_matched = []
    idx = 0
    for t_m in gt_cmp_t_matched:
        while gt_cmp_t[idx] != t_m:
            idx += 1
        gt_cmp_poses_matched.append(nc_gt_cmp[idx][1])
        idx += 1

    assert len(gt_cmp_poses_matched) == len(gt_cmp_t_matched)

    gt_cmp_matched = list(zip(gt_cmp_t_matched, gt_cmp_poses_matched))

    return gt_matched, gt_cmp_matched


def reduce_active_beams(ls: client.LidarScan, beams_num: int):
    """Reduces the active beams of a lidar scan to beams_num.
    
    Achieved by setting RANGE field to 0 of 'inactive' beams. 
    
    Args:
        beams_num: number of uniformaly distributed active beams
    """
    beam_idxs = np.linspace(0, ls.h, num=beams_num, endpoint=False, dtype=int)
    clean_mask = np.ones(ls.h, dtype=bool)
    clean_mask[beam_idxs] = 0
    # clearing only range, because for all downstream processing tasks
    # it's usually enough to not look at any other pixels
    ls.field(client.ChanField.RANGE)[clean_mask, :] = 0


def pose_scans_from_nc_gt(
    source,
    nc_gt_poses_file: Optional[str] = None,
    nc_gt_poses: Optional[List[Tuple[float, pu.PoseH]]] = None
):
    """Add poses to LidarScans stream Newer College Dataset ground truth poses.

    Scans with timestamps outside of the timestamps in `nc_gt_poses` file are
    skipped with a warning printed to stdout.

    Args:
        source: one of:
            - Sequence[client.LidarScan] - single scan sources
        nc_gt_poses_file: path to the file with in Newer College Dataset GT poses
                          format
        nc_gt_poses: list of tuples (ts, pose)
    """
    # load one pose per scan
    if nc_gt_poses_file:
        gts = read_newer_college_gt(nc_gt_poses_file)
    elif nc_gt_poses is not None:
        gts = nc_gt_poses

    # # make time indexed poses starting from 0.5
    traj_eval = pu.TrajectoryEvaluator(gts, time_bounds=1.5)

    skipped_scans = 0

    for obj in source:
        if isinstance(obj, client.LidarScan):
            # Iterator[client.LidarScan]
            scan = obj

            col_ts = scan.timestamp * 1e-9

            try:
                traj_eval(scan, col_ts=col_ts)
            except ValueError as e:
                skipped_scans += 1
                continue
            except AssertionError as e:
                print("WARNING (BROKEN SCANS?): ", str(e))
                skipped_scans += 1
                continue

            yield scan

    print(f"NOTE: Therere where {skipped_scans} skipped scans that wasn't "
          "because they were outside of the NC GT poses available")
