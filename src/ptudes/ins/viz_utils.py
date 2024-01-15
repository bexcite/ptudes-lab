from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import ouster.sdk.pose_util as pu
from scipy.spatial.transform import Rotation

from ptudes.ins.data import NavState, blk

from ptudes.viz_utils import (RED_COLOR, BLUE_COLOR, YELLOW_COLOR, GREY_COLOR,
                              GREY1_COLOR, WHITE_COLOR)


def lio_ekf_graphs(lio_ekf,
                   gt: Optional[Tuple[List, List]] = None,
                   gt2: Optional[Tuple[List, List]] = None,
                   labels: List[str] = []):
    """Plots of imus (mainly) logs"""

    # print("total_imu: accel = ",
    #       lio_ekf._imu_total.lacc / lio_ekf._imu_total_cnt + lio_ekf._g_fn)
    # print("total_imu: avel  = ",
    #       lio_ekf._imu_total.avel / lio_ekf._imu_total_cnt)
    # print(f"total_cnt = ", lio_ekf._imu_total_cnt)

    min_ts = lio_ekf._lg_t[0]
    # print(f"imu_ts: {min_ts}")
    if lio_ekf._nav_scan_idxs:
        nav_idx = lio_ekf._nav_scan_idxs[0]
        # scan_ts = scan_begin_ts(lio_ekf._navs[nav_idx].scan)
        scan_ts = lio_ekf._navs_t[nav_idx]
        # print(f"scan_ts: {scan_ts}")
        min_ts = min(min_ts, scan_ts)
    # print(f"min_ts res: {min_ts}")

    t = [t - min_ts for t in lio_ekf._lg_t]
    acc_x = [a[0] for a in lio_ekf._lg_acc]
    acc_y = [a[1] for a in lio_ekf._lg_acc]
    acc_z = [a[2] for a in lio_ekf._lg_acc]

    gyr_x = [a[0] for a in lio_ekf._lg_gyr]
    gyr_y = [a[1] for a in lio_ekf._lg_gyr]
    gyr_z = [a[2] for a in lio_ekf._lg_gyr]

    ba_x = [nav.bias_acc[0] for nav in lio_ekf._navs]
    ba_y = [nav.bias_acc[1] for nav in lio_ekf._navs]
    ba_z = [nav.bias_acc[2] for nav in lio_ekf._navs]

    bg_x = [nav.bias_gyr[0] for nav in lio_ekf._navs]
    bg_y = [nav.bias_gyr[1] for nav in lio_ekf._navs]
    bg_z = [nav.bias_gyr[2] for nav in lio_ekf._navs]

    nav_t = [nav_t - min_ts for nav_t in lio_ekf._navs_t]

    dpos = [nav.pos for nav in lio_ekf._navs]
    dpos_x = [p[0] for p in dpos]
    dpos_y = [p[1] for p in dpos]
    dpos_z = [p[2] for p in dpos]

    # Create the plot
    fig = plt.figure()
    ax = [
        plt.subplot(6, 2, 1),
        plt.subplot(6, 2, 3),
        plt.subplot(6, 2, 5),
        plt.subplot(6, 2, 7),
        plt.subplot(6, 2, 9),
        plt.subplot(6, 2, 11)
    ]
    for a in ax[1:]:
        a.sharex(ax[0])
    for a in ax[1:3]:
        a.sharey(ax[0])
    for a in ax[4:]:
        a.sharey(ax[3])
    for a in ax[:-1]:
        plt.setp(a.get_xticklines(), visible=False)
        plt.setp(a.get_xticklabels(), visible=False)

    i = 0
    ax[i].plot(t, acc_x, label="data (acc/gyr)")
    ax[i].plot(nav_t, ba_x, label="bias (acc/gyr)")
    ax[i].legend(frameon=False)
    ax[i].set_ylabel('acc_X')
    ax[i + 1].plot(t, acc_y)
    ax[i + 1].plot(nav_t, ba_y)
    ax[i + 1].set_ylabel('acc_Y')
    ax[i + 2].plot(t, acc_z)
    ax[i + 2].plot(nav_t, ba_z)
    ax[i + 2].set_ylabel('acc_Z')

    i = 3
    ax[i].plot(t, gyr_x)
    ax[i].plot(nav_t, bg_x)
    ax[i].set_ylabel('gyr_X')
    ax[i + 1].plot(t, gyr_y)
    ax[i + 1].plot(nav_t, bg_y)
    ax[i + 1].set_ylabel('gyr_Y')
    ax[i + 2].plot(t, gyr_z)
    ax[i + 2].plot(nav_t, bg_z)
    ax[i + 2].set_ylabel('gyr_Z')
    ax[i + 2].set_xlabel('t')

    axX = fig.add_subplot(6, 2, (2, 4))
    main_label = "resulting pose" if not labels else labels[0]
    axX.plot(nav_t, dpos_x, label=main_label)
    axX.legend(frameon=False)
    axX.grid(True)
    axY = fig.add_subplot(6, 2, (6, 8))
    axY.plot(nav_t, dpos_y)
    axY.grid(True)
    axZ = fig.add_subplot(6, 2, (10, 12))
    axZ.plot(nav_t, dpos_z)
    axZ.grid(True)
    axZ.set_xlabel('t')

    def draw_gt(gtX, line_label: str = ""):
        if gtX is not None and len(gtX[0]):
            gt_t = np.array(gtX[0]) - min_ts
            gt_poses = np.array(gtX[1])
            axX.plot(gt_t, gt_poses[:, 0, 3], label=line_label)
            if line_label:
                axX.legend(frameon=False)
            axY.plot(gt_t, gt_poses[:, 1, 3])
            axZ.plot(gt_t, gt_poses[:, 2, 3])

    draw_gt(gt, line_label=labels[1] if len(labels) > 1 else "gt compare 1")
    draw_gt(gt2, line_label=labels[2] if len(labels) > 2 else "gt compare 2")
    
    for a in ax + [axX, axY, axZ]:
        a.grid(True)

    # Draw navs/gt poses knots on t axis
    # scan_t = [lio_ekf._navs_t[si] - min_ts for si in lio_ekf._nav_scan_idxs]
    # for a in ax + [axX, axY, axZ]:
    #     a.plot(scan_t, np.zeros_like(scan_t), '8r')

    # if gt2 is not None and len(gt2[0]):
    #     gt2_t = np.array(gt2[0]) - min_ts
    #     for a in ax + [axX, axY, axZ]:
    #         a.plot(gt2_t, np.ones_like(gt2_t), '8b')

    plt.show()


def euler_angles_diff(nav1: NavState, nav2: NavState) -> np.ndarray:
    eul1 = Rotation.from_matrix(nav1.att_h).as_euler("XYZ")
    eul2 = Rotation.from_matrix(nav2.att_h).as_euler("XYZ")
    diff = (eul1 - eul2)
    diff[diff >= np.pi] -= 2 * np.pi
    diff[diff < -np.pi] += 2 * np.pi
    return diff


def lio_ekf_error_graphs(lio_ekf_gt, lio_ekf, lio_ekf_dr=None):
    """Plots of pos/angle errors"""

    assert set(lio_ekf_gt._navs_t) == set(lio_ekf._navs_t)

    dr_present = False
    if lio_ekf_dr is not None:
        dr_present = True
        assert set(lio_ekf_gt._navs_t) == set(lio_ekf_dr._navs_t)

    # remove navs dupes due to update navs in the list
    navs_gt = []
    navs = []
    navs_dr = []
    navs_t = sorted(list(set(lio_ekf_gt._navs_t)))
    ngt_it = iter(zip(lio_ekf_gt._navs_t[::-1], lio_ekf_gt._navs[::-1]))
    n_it = iter(zip(lio_ekf._navs_t[::-1], lio_ekf._navs[::-1]))
    if dr_present:
        ndr_it = iter(zip(lio_ekf_dr._navs_t[::-1], lio_ekf_dr._navs[::-1]))
    try:
        ngt_t, ngt = next(ngt_it)
        n_t, n = next(n_it)
        if dr_present:
            ndr_t, ndr = next(ndr_it)
        for t in navs_t[::-1]:
            assert t == ngt_t
            assert t == n_t
            navs_gt.append(ngt)
            navs.append(n)
            if dr_present:
                assert t == ndr_t
                navs_dr.append(ndr)
                while t == ndr_t:
                    ndr_t, ndr = next(ndr_it)
            while t == ngt_t:
                ngt_t, ngt = next(ngt_it)
            while t == n_t:
                n_t, n = next(n_it)
    except StopIteration:
        pass
    navs_gt = navs_gt[::-1]
    navs = navs[::-1]
    navs_dr = navs_dr[::-1]
    assert len(navs) == len(navs_t)
    assert len(navs) == len(navs_gt)
    if dr_present:
        assert len(navs) == len(navs_dr)

    min_ts = navs_t[0]

    dpos = np.array(
        [nav_gt.pos - nav.pos for nav_gt, nav in zip(navs_gt, navs)])
    dpos_x, dpos_y, dpos_z = np.hsplit(dpos, 3)

    deul = np.array(
        [euler_angles_diff(nav_gt, nav) for nav_gt, nav in zip(navs_gt, navs)])
    deul_r, deul_p, deul_y = np.hsplit(deul, 3)

    if dr_present:
        drpos = np.array(
            [nav_gt.pos - nav.pos for nav_gt, nav in zip(navs_gt, navs_dr)])
        drpos_x, drpos_y, drpos_z = np.hsplit(drpos, 3)

        dreul = np.array([
            euler_angles_diff(nav_gt, nav)
            for nav_gt, nav in zip(navs_gt, navs_dr)
        ])
        dreul_r, dreul_p, dreul_y = np.hsplit(dreul, 3)

    fig, ax = plt.subplots(6, 1, sharex=True)
    for a in ax.flat:
        a.grid(True)

    upd_t = [lio_ekf._navs_t[si] - min_ts for si in lio_ekf._nav_scan_idxs]

    t = [t - min_ts for t in navs_t]

    i = 0
    ax[i].plot(t, dpos_x, label='error between gt and corrected')
    ax[i].set_ylabel('X err')
    ax[i + 1].plot(t, dpos_y)
    ax[i + 1].set_ylabel('Y err')
    ax[i + 2].plot(t, dpos_z)
    ax[i + 2].set_ylabel('Z err')
    if dr_present:
        ax[i].plot(t, drpos_x, label='error between gt and dr')
        ax[i + 1].plot(t, drpos_y)
        ax[i + 2].plot(t, drpos_z)

    i = 3
    ax[i].plot(t, deul_r)
    ax[i].set_ylabel('Roll err')
    ax[i + 1].plot(t, deul_p)
    ax[i + 1].set_ylabel('Pitch err')
    ax[i + 2].plot(t, deul_y)
    ax[i + 2].set_ylabel('Yaw err')
    if dr_present:
        ax[i].plot(t, dreul_r)
        ax[i + 1].plot(t, dreul_p)
        ax[i + 2].plot(t, dreul_y)


    for idx, a in enumerate(ax.flat):
        a.plot(upd_t,
               np.zeros_like(upd_t),
               '8r',
               label="correction (update)" if idx == 0 else "")

    fig.legend(loc="upper center")

    plt.show()


def lio_ekf_viz(lio_ekf):
    """Visualize 3D poses and nav state histories"""

    import ouster.viz as viz
    from ptudes.utils import (make_point_viz)
    from ptudes.viz_utils import PointCloud
    point_viz = make_point_viz(f"Traj: poses = {len(lio_ekf._navs)}",
                               show_origin=True)

    def next_scan_based_nav(navs: List[NavState], start_idx: int = 0) -> int:
        ln = len(navs)
        start_idx = (start_idx + ln) % ln
        curr_idx = start_idx
        while navs[curr_idx].scan is None:
            curr_idx = (curr_idx + 1) % ln
            if curr_idx == start_idx:
                break
        return curr_idx

    @dataclass
    class CloudsStruct:
        xyz: PointCloud
        frame: PointCloud
        frame_ds: PointCloud
        src_source: PointCloud
        src_source_hl: PointCloud
        src: PointCloud
        tgt: PointCloud
        src_hl: PointCloud
        tgt_hl: PointCloud
        local_map: PointCloud
        kiss_map: PointCloud

        def __init__(self, point_viz: viz.PointViz):
            self.xyz = PointCloud(point_viz,
                                  point_color=GREY_COLOR,
                                  point_size=1)
            self.frame = PointCloud(point_viz,
                                    point_color=GREY1_COLOR,
                                    point_size=1)
            self.frame_ds = PointCloud(point_viz,
                                       point_color=GREY1_COLOR,
                                       point_size=2)

            self.src_source = PointCloud(point_viz)
            self.src_source_hl = PointCloud(point_viz,
                                            point_color=YELLOW_COLOR,
                                            point_size=5)
            self.src = PointCloud(point_viz)
            self.src_hl = PointCloud(point_viz,
                                     point_color=RED_COLOR,
                                     point_size=5)
            self.tgt = PointCloud(point_viz)
            self.tgt_hl = PointCloud(point_viz,
                                     point_color=BLUE_COLOR,
                                     point_size=5)
            self.local_map = PointCloud(point_viz)
            self.kiss_map = PointCloud(point_viz, point_color=WHITE_COLOR)

        def toggle(self):
            self.xyz.toggle()
            self.frame.toggle()
            self.frame_ds.toggle()
            self.src_source.toggle()
            self.src_source_hl.toggle()
            self.src.toggle()
            self.src_hl.toggle()
            self.tgt.toggle()
            self.tgt_hl.toggle()
            self.local_map.toggle()
            self.kiss_map.toggle()

        def disable(self):
            self.xyz.disable()
            self.frame.disable()
            self.frame_ds.disable()
            self.src_source.disable()
            self.src_source_hl.disable()
            self.src.disable()
            self.src_hl.disable()
            self.tgt.disable()
            self.tgt_hl.disable()
            self.local_map.disable()
            self.kiss_map.disable()

    clouds: Dict[int, CloudsStruct] = dict()

    sample_cloud = PointCloud(point_viz,
                              point_size=1,
                              point_color=YELLOW_COLOR)

    def set_cloud_from_idx(idx: int):
        nonlocal clouds
        nav = lio_ekf._navs[idx]
        if nav.scan is None:
            return
        if idx not in clouds:
            clouds[idx] = CloudsStruct(point_viz)
            clouds[idx].xyz.pose = nav.pose_mat()
            clouds[idx].xyz.points = nav.xyz
            clouds[idx].frame.pose = nav.pose_mat()
            clouds[idx].frame.points = nav.frame
            clouds[idx].frame_ds.pose = nav.pose_mat()
            clouds[idx].frame_ds.points = nav.frame_ds
            clouds[idx].src_source.pose = nav.pose_mat()
            clouds[idx].src_source.points = nav.src_source
            clouds[idx].src_source_hl.pose = nav.pose_mat()
            clouds[idx].src_source_hl.points = nav.src_source_hl
            clouds[idx].src.points = nav.src
            clouds[idx].src_hl.points = nav.src_hl
            clouds[idx].tgt.points = nav.tgt
            clouds[idx].tgt_hl.points = nav.tgt_hl
            clouds[idx].local_map.points = nav.local_map
            clouds[idx].kiss_map.points = nav.kiss_map
            clouds[idx].disable()
        clouds[idx].kiss_map.enable()

    def toggle_cloud_from_idx(idx: int, atr: Optional[str] = None):
        nonlocal clouds
        if idx not in clouds:
            return
        if atr is None or not hasattr(clouds[idx], atr):
            clouds[idx].toggle()
        elif hasattr(clouds[idx], atr):
            cloud = getattr(clouds[idx], atr)
            print(f"toggle {atr}[{idx}], size = ", cloud.points.shape)
            cloud.toggle()

    nav_axis_imu: List[viz.AxisWithLabel] = []
    nav_axis_scan: List[viz.AxisWithLabel] = []
    nav_axis_kiss: List[viz.AxisWithLabel] = []

    # bag_file = "~/data/newer-college/2021-ouster-os0-128-alphasense/collection 1 - newer college/2021-07-01-10-37-38-quad-easy.bag"
    # from ptudes.bag import IMUBagSource
    # from itertools import islice
    # imu_source = IMUBagSource(bag_file, imu_topic="/alphasense_driver_ros/imu")
    # imus = list(islice(imu_source, 100))
    # gts = read_newer_college_gt("~/data/newer-college/2021-ouster-os0-128-alphasense/collection 1 - newer college/ground_truth/gt-nc-quad-easy.csv")
    # gt_ts = gts[0][0]
    # gt_pose = gts[0][1]
    # for idx, im in enumerate(imus):
    #     print(f"dt[{idx}] = ", gts[0][0] - im.ts)

    # print("dts = ", imus[0].ts, gts[0][0], gts[0][0] - imus[0].ts)
    # print("gt_pose = ", gt_pose)

    # for t, p in gts:
    #     viz.AxisWithLabel(point_viz, pose=np.linalg.inv(gt_pose) @ p)

    sample_axis: List[viz.AxisWithLabel] = []
    for i in range(100):
        sample_axis += [
            viz.AxisWithLabel(point_viz, length=0.2, thickness=1, enabled=True)
        ]

    for idx, nav in enumerate(lio_ekf._navs):
        pose_mat = nav.pose_mat()
        axis_length = 0.5 if nav.update is not None else 0.1
        # axis_label = f"{idx}" if nav.scan is not None else ""
        axis_label = ""
        if nav.update:
            nav_axis_scan += [
                viz.AxisWithLabel(point_viz,
                                  pose=pose_mat,
                                  length=axis_length,
                                  label=axis_label)
            ]
            if nav.kiss_pose is not None:
                nav_axis_kiss += [
                    viz.AxisWithLabel(point_viz,
                                      pose=nav.kiss_pose,
                                      length=axis_length)
                ]
        else:
            nav_axis_imu += [
                viz.AxisWithLabel(point_viz,
                                  pose=pose_mat,
                                  length=axis_length,
                                  label=axis_label)
            ]

    def toggle_things(objs):
        for o in objs:
            if hasattr(o, "toggle"):
                o.toggle()

    def show_pos_att_uncertainty(nav, cov):
        pos_cov = cov[:3, :3]
        pos_m = nav.pos
        points = np.random.multivariate_normal(pos_m, pos_cov, size=2000)

        sample_cloud.points = points

        rot_cov = blk(cov, lio_ekf.PHI_ID, lio_ekf.PHI_ID, 3)

        es = np.random.multivariate_normal(mean=np.zeros(3),
                                           cov=rot_cov,
                                           size=len(sample_axis))
        for idx, e in enumerate(es):
            ax_pose = np.eye(4)
            ax_pose[:3, :3] = pu.exp_rot_vec(e) @ nav.att_h
            ax_pose[:3, 3] = nav.pos
            sample_axis[idx].pose = ax_pose
            sample_axis[idx].enable()

    target_id = 0
    target_idx = lio_ekf._nav_scan_idxs[target_id]

    set_cloud_from_idx(target_idx)

    def handle_keys(ctx, key, mods) -> bool:
        nonlocal target_idx, target_id, sample_axis
        if key == 32:
            if target_idx in clouds:
                clouds[target_idx].disable()
            target_id = (target_id + 1) % len(lio_ekf._nav_scan_idxs)
            target_idx = lio_ekf._nav_scan_idxs[target_id]

            target_nav = lio_ekf._navs[target_idx]
            print(f"TNAV[{target_idx}]: ", target_nav)
            # point_viz.camera.set_target(
            #     np.linalg.inv(target_nav.pose_mat()))
            point_viz.camera.set_target(np.linalg.inv(target_nav.kiss_pose))
            set_cloud_from_idx(target_idx)
            pred_nav = lio_ekf._navs_pred[target_idx]
            show_pos_att_uncertainty(pred_nav, pred_nav.cov)
            point_viz.update()
        elif key == ord('V'):
            toggle_cloud_from_idx(target_idx)
            point_viz.update()
        elif key == ord('G'):
            toggle_cloud_from_idx(target_idx, "src_source")
            toggle_cloud_from_idx(target_idx, "src_source_hl")
            point_viz.update()
        elif key == ord('H'):
            toggle_cloud_from_idx(target_idx, "src")
            toggle_cloud_from_idx(target_idx, "src_hl")
            point_viz.update()
        elif key == ord('J'):
            toggle_cloud_from_idx(target_idx, "tgt")
            toggle_cloud_from_idx(target_idx, "tgt_hl")
            point_viz.update()
        elif key == ord('T'):
            toggle_cloud_from_idx(target_idx, "xyz")
            point_viz.update()
        elif key == ord('Y'):
            toggle_cloud_from_idx(target_idx, "frame")
            point_viz.update()
        elif key == ord('U'):
            toggle_cloud_from_idx(target_idx, "frame_ds")
            point_viz.update()
        elif key == ord('M'):
            if mods == 0:
                toggle_cloud_from_idx(target_idx, "local_map")
            elif mods == 2:
                toggle_cloud_from_idx(target_idx, "kiss_map")
            point_viz.update()
        elif key == ord('P'):
            print(f"{mods = }")
            if mods == 0:
                toggle_things(nav_axis_imu)
            elif mods == 1:
                toggle_things(nav_axis_scan)
            elif mods == 2:
                toggle_things(nav_axis_kiss)
            point_viz.update()
        elif key == 91:  # "{"
            # sample points from distribution
            # marginalize on pos (and eevrything at mean)
            # after prediction (pre update from lidar scan)
            pred_nav = lio_ekf._navs_pred[target_idx]
            # for idx, pn in enumerate(lio_ekf._navs):
            #     print(f"_navs[{idx}] = ", pn)
            # for idx, pn in enumerate(lio_ekf._navs_pred):
            #     print(f"navs_pred[{idx}] = ", pn)
            assert len(lio_ekf._navs) == len(lio_ekf._navs_pred)
            print(f"pred_nav[{target_idx}] = ", pred_nav)
            show_pos_att_uncertainty(pred_nav, pred_nav.cov)
            # print("nav_scan_idxs = ", lio_ekf._nav_scan_idxs)
            print("target_idx = ", target_idx)
            point_viz.update()
            #
        elif key == 93:  # "}"
            # after update
            upd_nav = lio_ekf._navs[target_idx]
            print(f"upd_nav[{target_idx}] = ", upd_nav)
            show_pos_att_uncertainty(upd_nav, upd_nav.cov)
            point_viz.update()
        else:
            print("key = ", key)
        return True

    point_viz.push_key_handler(handle_keys)

    target_nav = lio_ekf._navs[target_idx]
    point_viz.camera.set_target(np.linalg.inv(target_nav.pose_mat()))

    point_viz.update()
    point_viz.run()
