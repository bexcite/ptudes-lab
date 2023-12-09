"""Ouster lio odometry experiments"""

import click
from typing import Optional

import ouster.client as client
from ouster.sdk.util import resolve_metadata

from ptudes.utils import (read_metadata_json, read_packet_source)
from ptudes.lio_ekf import LioEkfScans

from ouster.viz import PointViz, LidarScanViz, SimpleViz

@click.command(name="odom")
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
def ptudes_odom(file: str, meta: Optional[str]) -> None:
    """Ouster Lidar Odometry"""

    meta = resolve_metadata(file, meta)
    if not meta:
        raise click.ClickException(
            "File not found, please specify a metadata file with `-m`")
    click.echo(f"Reading metadata from: {meta}")
    info = read_metadata_json(meta)

    packet_source = read_packet_source(file, meta=info)
    odom_scans = LioEkfScans(packet_source)

    for ls in odom_scans:
        print("==" * 30)

    # scans = iter(scans_source)

    # rates = SimpleViz._playback_rates
    # if rate not in rates:
    #     click.echo(
    #         f"WARNING: {rate = } is not found in {rates}, using rate = 1.0")
    #     rate = 1.0

    # point_viz = PointViz("Ptudes Viz")
    # ls_viz = LidarScanViz(scans_source.metadata, viz=point_viz)
    # SimpleViz(ls_viz, rate=rate, on_eof="stop").run(scans)
