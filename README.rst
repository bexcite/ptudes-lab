=========================================================
Ptudes Lab: Lidar odometry, SLAM and visualization tools
=========================================================

This is a playground of various experiments with SLAM, mapping and visualization
of lidar point clouds.

Table of contents:

- `Ouster SDK/Kiss-ICP flyby viz <flyby-viz>`_

.. _flyby-viz:

Ouster Lidar/SDK flyby visualizer using odometry/slam poses (Kiss-ICP based)
----------------------------------------------------------------------------

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

1. Get Ouster sensor lidar data in a ``.pcap`` format
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

.. _official sensor docs: https://static.ouster.dev/sensor-docs/#sample-data
.. _KISS-ICP: https://github.com/PRBonn/kiss-icp

How to run:
~~~~~~~~~~~

Once you have Ouster ``.pcap`` data and poses per every scan in kitti format you
can run ``flyby`` command as::

    ptudes flyby ./OS-0-128_v3.0.1_1024x10.pcap --kitti-poses ./results/latest/OS-0-128_v3.0.1_poses_kitti.txt

Use ``--help`` to see more options like ``--start-scan/--end-scan`` to view only
a specific range of scans.

Some useful keybord shortcuts:

    ==============  =============================================================
        Key         Action
    ==============  =============================================================
    ``SPACE``       Stop/Start flying
    ``>``           Increase/decrease flying speed
    ``8``           Toggle poses/trajectory view
    ``k / K``       Cycle point cloud coloring mode of accumulated clouds or map
    ``g / G``       Cycle point cloud color palette of accumulated clouds or map
    ``j / J``       Increase/decrease point size of accumulated clouds or map
    ==============  =============================================================

