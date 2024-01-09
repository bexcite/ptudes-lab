from typing import Optional, List, Union, Iterator

import numpy as np

from pathlib import Path

import time
from rosbags.highlevel import AnyReader

from ptudes.ins.data import IMU

import ouster.client as client
from ouster.client import UDPProfileLidar

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


class IMUBagSource:
    """Read imu msgs from ROS bags"""

    def __init__(self, data_path: Union[str, list],
                 imu_topic: Optional[str] = None):

        if isinstance(data_path, list):
            data = [Path(p) for p in data_path]
        else:
            data = [Path(data_path)]

        self._bag_reader = AnyReader(data)
        self._bag_reader.open()

        self._conns = []

        imu_conns = [
            c for c in self._bag_reader.connections
            if (c.msgtype == "sensor_msgs/msg/Imu" or (
                c.msgtype == "ouster_ros/msg/PacketMsg"
                and c.topic.endswith("imu_packets")))
        ]
        assert len(imu_conns), "Expect any topic with msgtype: " \
            "sensor_msgs/msg/Imu or Ouster imu_packets types but found None"
        if imu_topic is not None:
            self._conns += [c for c in imu_conns if c.topic == imu_topic]
            assert len(self._conns), "Expect a topic with msgtype: " \
                f"sensor_msgs/msg/Imu and '{imu_topic}' name but found None"
        else:
            self._conns += [imu_conns[0]]


    def __iter__(self) -> Iterator[IMU]:

        # since imu packets haven't changed since original Ouster
        # packets form we can use any sane packet format to parse them
        _pf = client._client.PacketFormat.from_profile(
            UDPProfileLidar.PROFILE_LIDAR_RNG19_RFL8_SIG16_NIR16, 64, 16)

        for conn, ts, rawdata in self._bag_reader.messages(
                connections=self._conns):

            msg = self._bag_reader.deserialize(rawdata, conn.msgtype)

            if conn.msgtype == "sensor_msgs/msg/Imu":
                msg_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                lacc = np.array([
                    msg.linear_acceleration.x, msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ])
                avel = np.array([
                    msg.angular_velocity.x, msg.angular_velocity.y,
                    msg.angular_velocity.z
                ])
                yield IMU(lacc, avel, msg_ts)
            elif conn.msgtype == "ouster_ros/msg/PacketMsg":
                # Ouster imu packet decoding
                msg_ts = ts * 1e+9
                imu_packet = client.ImuPacket(msg.buf,
                                              timestamp=msg_ts,
                                              packet_format=_pf)
                yield IMU.from_packet(imu_packet)
