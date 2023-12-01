=========================================================
Ptudes Lab: Lidar odometry, SLAM and visualization tools
=========================================================

This is a playground of various experiments with SLAM, mapping and visualization
of lidar point clouds.

It's heavily using and relying on Ouster sensor lidar data using Ouster SDK, datasets that contain
Ouster lidar data and lidar odometry poses from `KISS-ICP`_ package.

Table of contents:

.. contents::
   :local:
   :depth: 1

.. _flyby-viz:

Flyby 3d visualizations of lidar data with poses
-------------------------------------------------

Review the registered point cloud map using the per scan poses of the
odometry/slam pipeline with deskewing and point coloring by ``REFLECTIVITY``,
``NEAR_IR``, ``SIGNAL`` and ``RANGE`` channels (channels availability depends on
the UDP Lidar Profile of the data).

.. figure:: https://github.com/bexcite/ptudes-lab/raw/main/docs/images/flyby.png

Pre-requisite:
~~~~~~~~~~~~~~

0. Installation
````````````````

You can install ``ptudes-lab`` from PyPi using::

    pip install git+https://github.com/bexcite/ptudes-lab.git@main#egg=ptudes-lab

or you can install directly from the source code repository::

    pip install .

1. Get Ouster sensor lidar data in a ``.pcap/.bag`` format
```````````````````````````````````````````````````````

You can download a sample data from the `official sensor docs`_:

Or you can record it from the sensor if you have one, using ``ouster-sdk/cli``::

    ouster-cli source <MY_SENSOR_IP> record

2. Get the lidar scans poses in kitti format
`````````````````````````````````````````````

All my experiments based on `KISS-ICP`_ poses outputs in kitti format. To get
the scan poses you can run ``kiss-icp`` pipeline on the previously obtained
Ouster ``.pcap`` data using::

    kiss_icp_pipeline --deskew ./OS-0-128_v3.0.1_1024x10.pcap

You can use any poses source with ``--kitti-poses`` in the command ``ptudes
flyby`` below and not neccessarily ``KISS-ICP`` output. For example it can be
the result of some post-processing step (smoothing, loop closure, fusion with
other sensors etc) the only requirement is that the number of poses should be
the same as the number of scans in the ``.pcap/.bag`` file.

.. _official sensor docs: https://static.ouster.dev/sensor-docs/#sample-data
.. _KISS-ICP: https://github.com/PRBonn/kiss-icp

How to run:
~~~~~~~~~~~

Once you have Ouster sensor ``.pcap/.bag`` data and poses per every scan in kitti
format you can run ``ptudes flyby`` command as::

    ptudes flyby ./OS-0-128_v3.0.1_1024x10.pcap --kitti-poses ./results/latest/OS-0-128_v3.0.1_poses_kitti.txt

or for example using ``.bag`` from `Newer College`_ dataset::

    ptudes flyby ./newer-college/2021-ouster-os0-128-alphasense/collection1/2021-07-01-10-37-38-quad-easy.bag \
        --meta ~/data/newer-college/2021-ouster-os0-128-alphasense//beam_intrinsics_os0-128.json \
        --kitti-poses ./2021-07-01-10-37-38-quad-easy_poses_kitti.txt \
        --start-scan 20 \
        --end-scan 50

Use ``--help`` to see more options like ``--start-scan/--end-scan`` to view only
a specific range of scans.

Some useful keybord shortcuts for ``flyby`` command:

==============  =============================================================
Key             Action
==============  =============================================================
``SPACE``       Stop/Start flying
``>``           Increase/decrease flying speed
``8``           Toggle poses/trajectory view
``k / K``       Cycle point cloud coloring mode of accumulated clouds or map
``g / G``       Cycle point cloud color palette of accumulated clouds or map
``j / J``       Increase/decrease point size of accumulated clouds or map
==============  =============================================================

.. _Newer College: https://ori-drs.github.io/newer-college-dataset/


ROS bags visualizations of raw lidar data
------------------------------------------------------

Ouster sensors produce raw ``lidar_packets/imu_packets`` data in a corresponding
ROS topics. To view the point cloud from such raw packets BAGs without spinning a
ROS and installing all drivers one can use ``ptudes viz`` command.

    Not tested with ROS2 bags:(
    
    I wasn't been able to locate the ROS2 bag with raw Ouster ``lidar_packets``,
    so if you by any chance have such a ROS2 bag that you can share with me I
    can make sure that both ROS1 and ROS2 bags working for the ``ptudes viz``
    command. (i.e. ``ptudes.bag.OusterRawBagSource`` packet source can work with
    ROS1/ROS2 bags)

For example to visualize `Newer College` dataset BAGS use::

    ptudes viz ./newer-college/2021-ouster-os0-128-alphasense/collection1/2021-07-01-10-37-38-quad-easy.bag \
        --meta ~/data/newer-college/2021-ouster-os0-128-alphasense//beam_intrinsics_os0-128.json

.. figure:: https://github.com/bexcite/ptudes-lab/raw/pb/bags-viz/docs/images/viz_nc_bag.png

Since the underlying Viz is the `PointViz`_ shipped with Ouster SDK the full
list of keyboard shortcuts can be found `here`_

.. _PointViz: https://static.ouster.dev/sdk-docs/python/viz/index.html
.. _here: https://static.ouster.dev/sdk-docs/sample-data.html#id1


