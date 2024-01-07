"""Error-state Extended Kalman Filter experiments"""

import click
from typing import Optional

import ouster.client as client
from ouster.sdk.util import resolve_metadata

from ptudes.utils import (read_metadata_json, read_packet_source)
from ptudes.lio_ekf import LioEkfScans


@click.group(name="ekf-bench")
def ptudes_ekf_bench() -> None:
    """ES EKF benchmarks and experiments.

    Various experiments with Error-state kalman filters and IMU mechanizations.
    """
    pass


@click.command(name="sim")
@click.option("--start-scan", type=int, default=0, help="Start scan number")
@click.option("--end-scan", type=int, help="End scan number, inclusive")
@click.option("-p",
              "--plot",
              required=False,
              type=str,
              help="Plotting option [graphs, point_viz]")
def ptudes_ekf_sim(start_scan: int,
                end_scan: Optional[int] = None,
                plot: Optional[str] = None) -> None:
    """Simulated IMU measurements"""

    print("SIM MEAS")


ptudes_ekf_bench.add_command(ptudes_ekf_sim)