from typing import Optional, Callable

import numpy as np

import weakref
import ouster.viz as viz
from ouster.viz import (PointViz, ScansAccumulator, add_default_controls)
import ouster.sdk.pose_util as pu


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
    # print(f"idx = {start_idx}")
    last_pose_inv = np.linalg.inv(pruned_poses[0][1])
    idx = start_idx + 1
    for tp in traj_poses[idx:end_idx+1]:
        p = tp[1]
        pd = pu.log_pose(last_pose_inv @ p)
        pda = np.linalg.norm(pd[:3])
        pdm = np.linalg.norm(pd[3:])
        if pda > min_dist_angle * np.pi / 180 or pdm > min_dist_m or idx == end_idx:
            # print(f"{idx = }, {pd = }")
            # print(f"{pda = }, {pdm = }")
            pruned_poses.append(tp)
            last_pose_inv = np.linalg.inv(p)
        idx += 1
    # HACK: handle edge case when just one scan is selected and add the next
    # one for traj evaluator to work ...
    if len(pruned_poses) < 2 and end_idx + 1 < len(traj_poses):
        pruned_poses.append(traj_poses[end_idx + 1])
        # print(f"idx = {end_idx + 1}")
    # print(f"{len(pruned_poses) = }, {len(traj_poses) = }")
    return pu.make_kiss_traj_poses([p for _, p in pruned_poses])
