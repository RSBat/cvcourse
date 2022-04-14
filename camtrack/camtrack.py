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
    TriangulationParameters,
    build_correspondences,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    triangulate_correspondences,
    view_mat3x4_to_pose, rodrigues_and_translation_to_view_mat3x4,
)


MAX_REPROJ_ERROR = 5.0
TRIANG_PARAMS = TriangulationParameters(max_reprojection_error=MAX_REPROJ_ERROR,
                                        min_triangulation_angle_deg=1,
                                        min_depth=0)


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

    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    cloud, id_triangulated, _ = triangulate_correspondences(correspondences, vm_1, vm_2, intrinsic_mat, TRIANG_PARAMS)
    point_cloud_builder = PointCloudBuilder(id_triangulated,
                                            cloud)

    view_mats = []
    for frame, corners in enumerate(corner_storage):
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
