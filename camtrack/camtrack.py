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
    view_mat3x4_to_pose
)


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
    f1_corners = corner_storage[known_view_1[0]]
    f2_corners = corner_storage[known_view_2[0]]
    id_inter, (idx_1, idx_2) = snp.intersect(f1_corners.ids.flatten(), f2_corners.ids.flatten(), indices=True)

    cloud = cv2.triangulatePoints(
        intrinsic_mat @ vm_1,
        intrinsic_mat @ vm_2,
        f1_corners.points[idx_1].T,
        f2_corners.points[idx_2].T,
    ).T
    cloud /= cloud[:, 3].reshape(-1, 1)
    cloud = cloud[:, :3]
    point_cloud_builder = PointCloudBuilder(id_inter,
                                            cloud)

    view_mats = []
    for frame, corners in enumerate(corner_storage):
        ids, (lhs, rhs) = snp.intersect(id_inter.flatten(), corners.ids.flatten(), indices=True)
        _, rvec, tvec = cv2.solvePnP(cloud[lhs].copy(), corners.points[rhs].copy(), intrinsic_mat, None)
        pose = Pose(cv2.Rodrigues(-rvec)[0], -cv2.Rodrigues(-rvec)[0] @ tvec)
        view_mats.append(pose_to_view_mat3x4(pose))

    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
