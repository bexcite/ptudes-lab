"""Ouster raw packets ROS Bags viz (mainly)"""

import click
from typing import Optional

import ouster.client as client
from ouster.sdk.util import resolve_metadata

from ptudes.utils import (read_metadata_json, read_packet_source)

from ouster.viz import PointViz, LidarScanViz, SimpleViz

@click.command(name="viz")
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
@click.option('-r',
              '--rate',
              type=float,
              default=1.0,
              help="Playback rate: 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0")
def ptudes_viz(file: str, meta: Optional[str],
               rate: float) -> None:
    """Ouster Lidar BAGS/PCAP 3d visualizer

    Data is provided via FILE in Ouster raw packets formats: PCAP or BAG with lidar/imu packets.

    For example to view the ROS bags with raw lidar packets like Newer College 2021
    dataset:

        ptudes viz BAG_FILE_OR_FOLDER -m PATH/beam_intrinsics.json

    NOTE: Only */[lidar,imu]_packets topics are replayed (not PointCloud2)
    """
    meta = resolve_metadata(file, meta)
    if not meta:
        raise click.ClickException(
            "File not found, please specify a metadata file with `-m`")
    print(f"Reading metadata from: {meta}")
    info = read_metadata_json(meta)

    packet_source = read_packet_source(file, meta=info)
    scans_source = client.Scans(packet_source)

    scans = iter(scans_source)

    rates = SimpleViz._playback_rates
    if rate not in rates:
        click.echo(
            f"WARNING: {rate = } is not found in {rates}, using rate = 1.0")
        rate = 1.0

    point_viz = PointViz("Ptudes Viz")
    ls_viz = LidarScanViz(scans_source.metadata, viz=point_viz)
    SimpleViz(ls_viz, rate=rate, on_eof="stop").run(scans)

    print(f"Scans produced: {scans_source._scans_produced}")
