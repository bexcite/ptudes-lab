from typing import List, Union, Iterator

from pathlib import Path

import time
from rosbags.highlevel import AnyReader

import ouster.client as client

# Adopted from Ouster SDK with changes to work on all platforms:
# https://github.com/ouster-lidar/ouster_example/blob/master/python/src/ouster/sdkx/bag.py

# Ouster ROS PacketMsg MD5 sum
OUSTER_PACKETMSG_MD5 = '4f7b5949e76f86d01e96b0e33ba9b5e3'

class OusterRawBagSource(client.PacketSource):
    """Read an Ouster raw sensor packet stream from ROS bag(s)"""
    _topics: List[str]
    _metadata: client.SensorInfo
    _rate: float
    _bag: AnyReader

    def __init__(self,
                data_path: Union[str, list],
                info: client.SensorInfo,
                *,
                rate: float = 0.0,
                lidar_topic: str = "",
                imu_topic: str = "") -> None:

        if isinstance(data_path, list):
            data = [Path(p) for p in data_path]
        else:
            data = [Path(data_path)]

        self._bag_reader = AnyReader(data)
        self._bag_reader.open()

        self._conns = []

        if not lidar_topic and not imu_topic:
            # Use any lidar/imu_packets topics if not set anything in ctor
            self._conns = [
                c for c in self._bag_reader.connections
                if c.topic.endswith("lidar_packets")
                or c.topic.endswith("imu_packets")
            ]
        else:
            topics = [t for t in [lidar_topic, imu_topic] if t]
            self._conns = [
                c for c in self._bag_reader.connections if c.topic in topics
            ]

        self._metadata = info
        self._rate = rate

    def __iter__(self) -> Iterator[client.Packet]:
        real_start_ts = time.monotonic()
        bag_start_ts = None
        for conn, ts, rawdata in self._bag_reader.messages(
                connections=self._conns):
            msg_ts_sec = ts / 10**9
            if self._rate:
                if not bag_start_ts:
                    bag_start_ts = msg_ts_sec
                real_delta = time.monotonic() - real_start_ts
                bag_delta = (msg_ts_sec -
                             bag_start_ts) / self._rate
                delta = max(0, bag_delta - real_delta)
                time.sleep(delta)

            if (conn.digest == OUSTER_PACKETMSG_MD5
                    and conn.topic.endswith("lidar_packets")):
                msg = self._bag_reader.deserialize(rawdata, conn.msgtype)
                yield client.LidarPacket(msg.buf, self._metadata, msg_ts_sec)

            elif (conn.digest == OUSTER_PACKETMSG_MD5
                    and conn.topic.endswith("imu_packets")):
                msg = self._bag_reader.deserialize(rawdata, conn.msgtype)
                yield client.ImuPacket(msg.buf, self._metadata, msg_ts_sec)

    @property
    def topics(self) -> List[str]:
        return [c.topic for c in self._conns]

    @property
    def metadata(self) -> client.SensorInfo:
        return self._metadata

    def close(self) -> None:
        self._bag_reader.close()
