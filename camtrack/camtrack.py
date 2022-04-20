#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp

from ba import run_bundle_adjustment
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    TriangulationParameters,
    build_correspondences,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    triangulate_correspondences,
    view_mat3x4_to_pose, rodrigues_and_translation_to_view_mat3x4, eye3x4, compute_reprojection_errors,
)


MAX_REPROJ_ERROR = 5.0
TRIANG_PARAMS = TriangulationParameters(max_reprojection_error=MAX_REPROJ_ERROR,
                                        min_triangulation_angle_deg=1,
                                        min_depth=0)


def recover_pose(corners_1, corners_2, intrinsic_mat) -> Optional[Pose]:
    correspondences = build_correspondences(corners_1, corners_2)
    correspondences_count = correspondences.ids.shape[0]
    print("Correspondences:", correspondences_count)
    print(corners_1.ids.shape[0], corners_2.ids.shape[0])
    # if correspondences_count < 0.8 * corners_1.ids.shape[0] or correspondences_count < 0.8 * corners_2.ids.shape[0]:
    #     return None

    E, mask = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2,
                                   intrinsic_mat, method=cv2.RANSAC)
    essential_mat_inliers_count = np.count_nonzero(mask)
    print("Inliers:", essential_mat_inliers_count)
    pose_inliers, R, t, pose_mask = cv2.recoverPose(E, correspondences.points_1, correspondences.points_2,
                                                    intrinsic_mat, mask=mask)
    # pose_inliers, E, R, t, mask = cv2.recoverPose(correnspondences.points_1, correnspondences.points_2,
    #                                         intrinsic_mat, None, intrinsic_mat, None, method=cv2.RANSAC)
    # if pose_inliers < 0.8 * correspondences_count:
    #     return None

    vm_1 = eye3x4()
    vm_2 = np.hstack([R, t])
    # vm_2 = pose_to_view_mat3x4(Pose(R, t))

    cloud, id_triangulated, avg_cos = triangulate_correspondences(correspondences, vm_1, vm_2,
                                                                  intrinsic_mat, TRIANG_PARAMS)

    print(correspondences.ids.shape, id_triangulated.shape)
    print(avg_cos)
    if id_triangulated.shape[0] < 100:  # id_triangulated.shape[0] < 0.8 * correspondences_count:
        return None

    return view_mat3x4_to_pose(vm_2)


def init_views(corner_storage, intrinsic_mat):
    for frame_1 in range(len(corner_storage) - 20):
        frame_2 = frame_1 + 20
        pose_2 = recover_pose(corner_storage[frame_1], corner_storage[frame_2], intrinsic_mat)

        if pose_2 is not None:
            pose_1 = view_mat3x4_to_pose(eye3x4())

            known_view_1 = frame_1, pose_1
            known_view_2 = frame_2, pose_2
            return known_view_1, known_view_2


def intersect_all(corner_storage: CornerStorage, frames: List[int]):
    intersection_ids = corner_storage[frames[0]].ids
    for frame in frames:
        intersection_ids = snp.intersect(intersection_ids.flatten(), corner_storage[frame].ids.flatten())

    points = []
    for frame in frames:
        _, (_, idx) = snp.intersect(intersection_ids.flatten(), corner_storage[frame].ids.flatten(), indices=True)
        points.append(corner_storage[frame].points[idx])

    return intersection_ids, points


def triangulate_multiple(corner_storage: CornerStorage, vms, intrinsic_mat, frames: List[int]):
    p3d_hom = []
    int_ids, int_points = intersect_all(corner_storage, frames)
    for pt_idx, _ in enumerate(int_ids):
        print(f"\r{pt_idx}/{len(int_ids)}", end="")
        eqs = []
        for points, vm in zip(int_points, vms):
            pt = points[pt_idx]
            pm = intrinsic_mat @ vm
            eqs.append(pt[0] * pm[2] - pm[0])
            eqs.append(pt[1] * pm[2] - pm[1])

        A = np.asarray(eqs)
        u, s, vh = np.linalg.svd(A)
        res = vh[-1]
        p3d_hom.append(res)
    print("\r", end="")

    p3d = cv2.convertPointsFromHomogeneous(np.asarray(p3d_hom)).reshape(-1, 3)

    mask = np.ones_like(int_ids).astype(bool)
    for points2d, vm in zip(int_points, vms):
        reproj_errs_1 = compute_reprojection_errors(p3d, points2d,
                                                    intrinsic_mat @ vm)
        mask = mask & (reproj_errs_1 < MAX_REPROJ_ERROR)
    print(f"Remaining: {np.count_nonzero(mask)}/{int_ids.shape[0]}")
    return int_ids[mask], p3d[mask]


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = init_views(corner_storage, intrinsic_mat)

    vm_1 = pose_to_view_mat3x4(known_view_1[1])
    vm_2 = pose_to_view_mat3x4(known_view_2[1])

    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    cloud, id_triangulated, _ = triangulate_correspondences(correspondences, vm_1, vm_2, intrinsic_mat, TRIANG_PARAMS)
    point_cloud_builder = PointCloudBuilder(id_triangulated,
                                            cloud)

    view_mats = []
    for frame, corners in enumerate(corner_storage):
        print(f"\rProcessing frame {frame + 1}/{len(corner_storage)}")
        _, (lhs, rhs) = snp.intersect(point_cloud_builder.ids.flatten(), corners.ids.flatten(), indices=True)
        _, rvec, tvec, _ = cv2.solvePnPRansac(point_cloud_builder.points[lhs].copy(), corners.points[rhs].copy(),
                                              intrinsic_mat, None)
        vm = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        view_mats.append(vm)

        if frame > 10:
            reference = max(0, frame - 10)
            correspondences = build_correspondences(corner_storage[reference], corners)
            new_cloud, new_triang_id, _ = triangulate_correspondences(correspondences, view_mats[reference], vm,
                                                                      intrinsic_mat, TRIANG_PARAMS)
            point_cloud_builder.add_only_new_points(new_triang_id, new_cloud)

        if frame >= 50:
            vms = [view_mats[frame - 10 * x] for x in range(6)]
            fns = [frame - 10 * x for x in range(6)]
            aa_ids, aa_pts = triangulate_multiple(corner_storage, vms, intrinsic_mat, fns)
            point_cloud_builder.add_points(aa_ids, aa_pts)

    # view_mats[known_view_1[0]] = vm_1
    # view_mats[known_view_2[0]] = vm_2

    # run_bundle_adjustment(
    #     intrinsic_mat,
    #     corner_storage,
    #     MAX_REPROJ_ERROR,
    #     view_mats,
    #     point_cloud_builder,
    # )

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        MAX_REPROJ_ERROR,
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
