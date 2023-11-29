=========================================================
Ptudes Lab: Lidar odometry, SLAM and visualization tools
=========================================================

This is a playground of various experiments with SLAM, mapping and visualization
of lidar point clouds.

Ouster Lidar/SDK flyby visualizer using odometry/slam poses (Kiss-ICP based)
----------------------------------------------------------------------------

Review the registered point cloud map using the per scan poses of the
odometry/slam pipeline with deskewing.

.. figure:: https://github.com/bexcite/ptudes-lab/raw/pb/readme-kick/docs/images/flyby.png

Pre-requisite:
~~~~~~~~~~~~~~

0. Installation

   You can install ``ptudes-lab`` from PyPi using::

      pip install ptudes-lab

   or you can install directly from the source code repository::

      pip install .

1. Get Ouster sensor lidar data in a ``.pcap`` format

   You can download a sample data from the `official sensor docs`_:

   Or you can record it from the sensor if you have one, using ``ouster-sdk/cli``::

      ouster-cli source <MY_SENSOR_IP> record

2. Get the lidar scans poses in kitti format

   All my experiments based on `KISS-ICP`_ outputs in kitti format. You can run
   ``kiss-icp`` pipeline on the previously obtained Ouster ``.pcap`` data using::

      kiss_icp_pipeline --deskew ./OS-0-128_v3.0.1_1024x10.pcap

.. _official sensor docs: https://static.ouster.dev/sensor-docs/#sample-data
.. _KISS-ICP: https://github.com/PRBonn/kiss-icp

How to run:
~~~~~~~~~~~

Once you have Ouster ``.pcap`` data and poses per every scan in kitti format you
can run ``flyby`` command as::

    ptudes flyby ./OS-0-128_v3.0.1_1024x10.pcap --kitti-poses ./results/latest/OS-0-128_v3.0.1_poses_kitti.txt

Some useful keybord shortcuts:

**TBD**