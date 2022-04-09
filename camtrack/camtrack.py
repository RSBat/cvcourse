#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    calc_inlier_indices,
)


MAX_REPROJ_ERROR = 5.0


def filter_reprojection(cloud_ids, cloud_points,
                        corner_ids, corner_points,
                        intrinsic_mat, view_mat):
    _, (cloud_idx, corner_idx) = snp.intersect(cloud_ids.flatten(), corner_ids.flatten(), indices=True)
    proj_mat = intrinsic_mat @ view_mat
    good_points = calc_inlier_indices(cloud_points[cloud_idx], corner_points[corner_idx], proj_mat, MAX_REPROJ_ERROR)
    return cloud_ids[cloud_idx][good_points], cloud_points[cloud_idx][good_points]


def triangulate(vm_1, vm_2, corners_1, corners_2, intrinsic_mat):
    ids, (idx_1, idx_2) = snp.intersect(corners_1.ids.flatten(), corners_2.ids.flatten(), indices=True)

    cloud = cv2.triangulatePoints(
        intrinsic_mat @ vm_1,
        intrinsic_mat @ vm_2,
        corners_1.points[idx_1].T,
        corners_2.points[idx_2].T,
    ).T
    cloud /= cloud[:, 3].reshape(-1, 1)
    cloud = cloud[:, :3]

    ids, cloud = filter_reprojection(ids, cloud, corners_1.ids, corners_1.points, intrinsic_mat, vm_1)
    ids, cloud = filter_reprojection(ids, cloud, corners_2.ids, corners_2.points, intrinsic_mat, vm_2)

    return ids, cloud


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    vm_1 = pose_to_view_mat3x4(known_view_1[1])
    vm_2 = pose_to_view_mat3x4(known_view_2[1])
    id_triangulated, cloud = triangulate(vm_1, vm_2,
                                         corner_storage[known_view_1[0]],
                                         corner_storage[known_view_2[0]],
                                         intrinsic_mat)
    point_cloud_builder = PointCloudBuilder(id_triangulated,
                                            cloud)

    view_mats = []
    for frame, corners in enumerate(corner_storage):
        ids, (lhs, rhs) = snp.intersect(point_cloud_builder.ids.flatten(), corners.ids.flatten(), indices=True)
        _, rvec, tvec, _ = cv2.solvePnPRansac(point_cloud_builder.points[lhs].copy(), corners.points[rhs].copy(),
                                              intrinsic_mat, None)
        pose = Pose(cv2.Rodrigues(-rvec)[0], -cv2.Rodrigues(-rvec)[0] @ tvec)

        vm = pose_to_view_mat3x4(pose)
        view_mats.append(vm)

        if frame > 10:
            reference = max(0, frame - 10)
            new_triang_id, new_cloud = triangulate(view_mats[reference], vm, corner_storage[reference],
                                                   corners, intrinsic_mat)
            point_cloud_builder.add_only_new_points(new_triang_id, new_cloud)

    view_mats[known_view_1[0]] = vm_1
    view_mats[known_view_2[0]] = vm_2

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
