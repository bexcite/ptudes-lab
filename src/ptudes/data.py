from typing import Optional, Union, Dict, Iterator

import ouster.client as client
import ouster.client._client as _client
from ouster.client import (PacketSource, UDPProfileLidar, FieldDType,
                           ChanField, LidarScan, LidarPacket, ImuPacket,
                           SensorInfo)

from ptudes.ins.data import IMU


class OusterLidarData:
    """Interface of PacketSource to LidarScan + IMU iterator"""

    def __init__(self,
                 source: PacketSource,
                 *,
                 fields: Optional[Dict[ChanField, FieldDType]] = None) -> None:
        """
        Args:
            source: any source of packets
            fields: chan field to add to LidarScans
        """
        self._source = source
        self._fields: Union[Dict[ChanField, FieldDType], UDPProfileLidar] = (
            fields if fields is not None else
            self._source.metadata.format.udp_profile_lidar)

    def __iter__(self) -> Iterator[Union[LidarScan, IMU]]:
        """Make an iterator."""

        w = self._source.metadata.format.columns_per_frame
        h = self._source.metadata.format.pixels_per_column
        columns_per_packet = self._source.metadata.format.columns_per_packet

        ls_write = None
        pf = _client.PacketFormat.from_info(self._source.metadata)
        batch = _client.ScanBatcher(w, pf)

        it = iter(self._source)
        while True:
            try:
                packet = next(it)
            except StopIteration:
                if ls_write is not None:
                    yield ls_write
                return

            if isinstance(packet, LidarPacket):
                ls_write = ls_write or LidarScan(h, w, self._fields,
                                                 columns_per_packet)
                if batch(packet, ls_write):
                    # Finished frame
                    yield ls_write
                    ls_write = None

            elif isinstance(packet, ImuPacket):
                yield IMU.from_packet(packet)

    def close(self) -> None:
        """Close the underlying PacketSource."""
        self._source.close()

    @property
    def metadata(self) -> SensorInfo:
        """Return metadata from the underlying PacketSource."""
        return self._source.metadata


def last_valid_packet_ts(scan: LidarScan) -> int:
    """Return last valid packet timestamp of a LidarScan"""
    columns_per_packet = scan.w // scan.packet_timestamp.shape[0]
    return scan.packet_timestamp[client.last_valid_column(scan) //
                                 columns_per_packet]
