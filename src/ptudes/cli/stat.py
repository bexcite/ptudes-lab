"""Ouster pcap/bag overall statistics"""

import click
from typing import Optional

import numpy as np
import ouster.client as client
from ouster.sdk.util import resolve_metadata

from ptudes.utils import (read_metadata_json, read_packet_source)

from ptudes.ins.data import StreamStatsTracker, IMU
from ptudes.data import OusterLidarData

@click.command(name="stat")
@click.argument(
    'file',
    required=True,
    type=click.Path(exists=True))
@click.option(
    '-m',
    '--meta',
    required=False,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Metadata for PCAP/BAG, required if automatic metadata resolution fails")
@click.option("--start-scan", type=int, default=0, help="Start scan number")
@click.option("--end-scan", type=int, help="End scan number, inclusive")
@click.option("--beams",
              type=int,
              default=0,
              help="Active beams number in a lidar scan (i.e. reduces "
              "beams (i.e. active rows of a scan) to the NUM)")
@click.option('-t',
              '--duration',
              type=float,
              default=3.0,
              help="Time period of the data (imu/scan) to read in seconds. "
              "(default: 3.0, use 0 to read and stats all data source)")
def ptudes_stat(file: str, meta: Optional[str],
                start_scan: int = 0,
                end_scan: Optional[int] = None,
                beams: int = 0,
                duration: float = 3) -> None:
    """Ouster BAGS/PCAP data source stats

    Calculates scans range and imu acc/gyr statistics for --duration seconds.
    
    Data is provided via FILE in Ouster raw packets formats: PCAP or BAG with
    lidar/imu packets.

    For example to view the Ouster imu/lidar packets stats on ROS bags from
    Newer College 2021 dataset:

        ptudes stat BAG_FILE_OR_FOLDER -m PATH/beam_intrinsics.json

    NOTE: Only */[lidar,imu]_packets topics are used for stats caclulation.
    """
    meta = resolve_metadata(file, meta)
    if not meta:
        raise click.ClickException(
            "File not found, please specify a metadata file with `-m`")
    print(f"Reading metadata from: {meta}")
    info = read_metadata_json(meta)

    packet_source = read_packet_source(file, meta=info)
    data_source = OusterLidarData(packet_source)

    stats = StreamStatsTracker(use_beams_num=beams,
                               metadata=data_source.metadata)

    scan_idx = 0

    for d in data_source:

        if isinstance(d, IMU):
            imu = d
            if scan_idx >= start_scan:
                stats.trackImu(d)


        elif isinstance(d, client.LidarScan):
            if scan_idx < start_scan:
                scan_idx += 1
                continue

            ls = d
            stats.trackScan(ls)
            scan_idx += 1

            if end_scan is not None and scan_idx > end_scan:
                break

        if duration and stats.dt > duration:
            break

    print(stats)
    grav_est = stats.acc_mean / np.linalg.norm(stats.acc_mean)
    print("Gravity vector estimation: ", grav_est)

