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

from ptudes.bag import OusterRawBagSource, IMUBagSource

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


def read_newer_college_gt(data_path: str) -> List[Tuple[float, np.ndarray]]:
    """Read ground truth poses for Newer College dataset"""
    gt_data = np.loadtxt(data_path, delimiter=",")
    ts = gt_data[:, 0] + gt_data[:, 1] * 1e-9

    pos = np.tile(np.eye(4), reps=(gt_data.shape[0], 1, 1))
    pos[:, :3, 3] = gt_data[:, 2:5]

    rots = Rotation.from_quat(gt_data[:, 5:9]).as_matrix()
    pos[:, :3, :3] = rots
    return [(t, p) for t, p in zip(ts[:], pos[:])]
