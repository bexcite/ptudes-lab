import click
from typing import Optional

import numpy as np

from ouster.cli.core import cli
import ouster.client as client
import ouster.pcap as pcap
from ouster.sdk.util import resolve_metadata
import ouster.sdk.pose_util as pu
from ouster.viz import (ScansAccumulator, Label)

import ouster.viz.scans_accum as scans_accum_module

from ptudes.utils import (make_point_viz, spin, estimate_apex_dolly,
                          map_points_num, prune_trajectory)

from ptudes.fly import (FlyingState, FState, BuildingState,
                        CameraTransitionState, CoursingState)

# max map/track cloud object sizes on init ("fixes" crash in ScansAccumulator)
scans_accum_module.MAP_INIT_POINTS_NUM = scans_accum_module.MAP_MAX_POINTS_NUM
scans_accum_module.TRACK_INIT_POINTS_NUM = scans_accum_module.TRACK_MAX_POINTS_NUM

@click.command(name="flyby")
@click.argument('file', required=True, type=click.Path(exists=True))
@click.option(
    '-m',
    '--meta',
    required=False,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Metadata for PCAP, helpful if automatic metadata resolution fails")
@click.option('--kitti-poses',
              required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              help="Poses file in Kitti format, one pose per scan "
              "(can be generated by kiss-icp)")
@click.option('-r',
              '--rate',
              type=float,
              default=1.0,
              help="Playback rate: 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0")
@click.option(
    "--accum-map-ratio",
    type=float,
    help="Ratio of random points of every scan to add to an overall map")
@click.option("--start-scan", type=int, default=0, help="Start scan number")
@click.option("--end-scan", type=int, help="End scan number, inclusive")
def ptudes_flyby(file: str, meta: Optional[str], kitti_poses: Optional[str],
                 rate: float, accum_map_ratio: Optional[float],
                 start_scan: int, end_scan: Optional[int]) -> None:
    """Show the flyby of the map."""
    meta = resolve_metadata(file, meta)
    if not meta:
        raise click.ClickException(
            "File not found, please specify a metadata file with `-m`")
    with open(meta) as json:
        click.echo(f"Reading metadata from: {meta}")
        info = client.SensorInfo(json.read())

    source = pcap.Pcap(file, info)
    scans_source = client.Scans(source)

    poses = pu.load_kitti_poses(kitti_poses)
    scans_num = poses.shape[0]

    scans = pu.pose_scans_from_kitti(scans_source, kitti_poses)

    start_scan = start_scan if start_scan < scans_num else scans_num - 1
    end_scan = (end_scan if end_scan is not None and end_scan < scans_num
                and end_scan >= start_scan else scans_num - 1)

    MAP_MAX_POINTS_NUM = 1500000
    # estimate accum map ratio for the densest map
    estimated_map_ratio = False
    if accum_map_ratio is None:
        # TODO: account for the column window when it's not 360
        pts_per_scan = info.format.pixels_per_column * info.format.columns_per_frame
        pts_total = (end_scan - start_scan + 1) * pts_per_scan
        accum_map_ratio = min(1.0, MAP_MAX_POINTS_NUM / pts_total)
        estimated_map_ratio = True
        click.echo(f"Estimated accum map ratio: {accum_map_ratio}")

    rates = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    try:
        rate_ind = rates.index(rate)
    except ValueError:
        click.echo(
            f"WARNING: {rate = } is not found in {rates}, using rate = 1.0")
        rate_ind = rates.index(1.0)

    point_viz = make_point_viz(title="Flyby")
    scans_accum = ScansAccumulator(scans_source.metadata,
                                   point_viz=point_viz,
                                   map_enabled=True,
                                   map_max_points=MAP_MAX_POINTS_NUM,
                                   map_select_ratio=accum_map_ratio)

    # initialize flyby osd
    flyby_osd = Label("", 1, 1, align_right=True)
    point_viz.add(flyby_osd)

    pause = False
    osd_enabled = True

    def handle_keys(ctx, key, mods) -> bool:
        nonlocal pause, rate_ind, osd_enabled
        if key == 32:
            pause = not pause
        elif key == ord('.') and mods == 1:
            rate_ind = (rate_ind + 1) % len(rates)
        elif key == ord(',') and mods == 1:
            rate_ind = (rate_ind + len(rates) - 1) % len(rates)
        elif key == ord('O'):
            osd_enabled = not osd_enabled
            scans_accum.toggle_osd(osd_enabled)
            scans_accum.draw(update=False)
        return True

    point_viz.push_key_handler(handle_keys)

    # min and max point in the cloud to calculate dolly
    min_max = np.zeros((3, 2))

    # prepare a pruned trajectory for camera
    traj_poses = pu.make_kiss_traj_poses(poses)
    pruned_traj_poses = prune_trajectory(traj_poses,
                                         start_idx=start_scan,
                                         end_idx=end_scan)
    coursing_traj_eval = pu.TrajectoryEvaluator(pruned_traj_poses,
                                                time_bounds=1.0)

    # the pose of the start_scan
    start_target = coursing_traj_eval._poses[0][1]

    def make_osd_str() -> str:
        pause_str = "" if not pause else " (PAUSED)"
        osd_str = f"map of scans: {start_scan} - {end_scan}"
        osd_str += f"\nmap num points: {map_points_num(scans_accum)} " + (
            "(O)" if scans_accum._map_overflow else "")
        osd_str += f"\nscans downsample ratio: {accum_map_ratio:.03f}"
        osd_str += f"\nstate: {fstate.name.replace('_', '..')}"
        osd_str += f"\nplayback: {rates[rate_ind]} {pause_str}"
        return osd_str

    def create_state(flying_state: FlyingState) -> FState:
        if flying_state == FlyingState.BUILDING:
            return BuildingState(scans,
                                 scans_accum,
                                 start_scan=start_scan,
                                 end_scan=end_scan,
                                 min_max=min_max,
                                 next_state=FlyingState.TO_THE_BEGINNING)
        elif flying_state == FlyingState.TO_THE_BEGINNING:
            return CameraTransitionState(name="TO_THE_BEGINNING",
                                         target=start_target,
                                         velocity=0.15,
                                         next_state=FlyingState.COURSING)
        elif flying_state == FlyingState.COURSING:
            return CoursingState(coursing_traj_eval,
                                 velocity=5.0,
                                 next_state=FlyingState.TO_THE_APEX)
        elif flying_state == FlyingState.TO_THE_APEX:
            dolly = estimate_apex_dolly(min_max, point_viz.camera.get_fov())
            return CameraTransitionState(name="TO_THE_APEX",
                                         pitch=0,
                                         yaw=140,
                                         dolly=dolly,
                                         velocity=0.15,
                                         next_state=FlyingState.TO_THE_BEGINNING)

    fstate: Optional[FState] = create_state(FlyingState.BUILDING)

    last_ts = -1
    def on_update(pviz, tick_ts) -> None:
        nonlocal last_ts, fstate, flyby_osd
        t = tick_ts

        # initialization step
        if last_ts < 0:
            last_ts = t
            return

        dt = tick_ts - last_ts
        dt = rates[rate_ind] * dt
        if pause:
            dt = 0
        last_ts = t

        new_state = fstate.update(dt, pviz)
        if new_state is not None:
            fstate = create_state(new_state)

        if osd_enabled:
            flyby_osd.set_text(make_osd_str())
        else:
            flyby_osd.set_text("")


    spin(point_viz,
         on_update,
         period=0.0333)
