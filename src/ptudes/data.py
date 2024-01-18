from typing import Optional, Union, Dict, Iterator, Tuple

import ouster.client as client
import ouster.client._client as _client
from ouster.client import (PacketSource, UDPProfileLidar, FieldDType,
                           ChanField, LidarScan, LidarPacket, ImuPacket,
                           SensorInfo)

from ptudes.ins.data import IMU


class OusterLidarData:
    """Lidar data source: LidarScan + IMUs iterator with scan index"""

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

        self._scan_idx = 0

    def withScanIdx(
        self,
        *,
        start_scan: int = 0,
        end_scan: Optional[int] = None
    ) -> Iterator[Tuple[int, Union[LidarScan, IMU]]]:
        """Make an iterator with (scanIdx, scan/imu)"""

        w = self._source.metadata.format.columns_per_frame
        h = self._source.metadata.format.pixels_per_column
        columns_per_packet = self._source.metadata.format.columns_per_packet

        ls_write = None
        pf = _client.PacketFormat.from_info(self._source.metadata)
        batch = _client.ScanBatcher(w, pf)

        scan_idx = 0

        it = iter(self._source)
        while True:
            try:
                packet = next(it)
            except StopIteration:
                if ls_write is not None:
                    yield scan_idx, ls_write
                    scan_idx += 1
                return

            if isinstance(packet, LidarPacket):
                ls_write = ls_write or LidarScan(h, w, self._fields,
                                                 columns_per_packet)
                if batch(packet, ls_write):
                    # Finished frame
                    # TODO: Scan batching op can be skipped when looking
                    # up for the start_scan (it will increase search speed)
                    if scan_idx >= start_scan:
                        yield scan_idx, ls_write
                    scan_idx += 1

                    if end_scan is not None and scan_idx > end_scan:
                        break

                    ls_write = None

            elif isinstance(packet, ImuPacket):
                if scan_idx >= start_scan:
                    yield scan_idx, IMU.from_packet(packet)

    def __iter__(self) -> Iterator[Tuple[int, Union[LidarScan, IMU]]]:
        """Make an iterator just data"""
        for scan_idx, d in self.withScanIdx():
            yield scan_idx, d

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
