#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import random
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
    for offset in [50, 30, 20]:
        for frame_1 in range(len(corner_storage) - offset):
            frame_2 = frame_1 + offset
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


def triangulate_multiple_frames(projections: List[np.ndarray], proj_mats: List[np.ndarray]) -> np.ndarray:
    """
    Triangulate using DLT
    :returns: triangulated point in homogeneous coordinates
    """
    eqs = []
    for projection, proj_mat in zip(projections, proj_mats):
        eqs.append(projection[0] * proj_mat[2] - proj_mat[0])
        eqs.append(projection[1] * proj_mat[2] - proj_mat[1])

    A = np.asarray(eqs)
    u, s, vh = np.linalg.svd(A)
    point3d_hom = vh[-1]
    return point3d_hom


def proj_hom(points3d_hom, proj_mat):
    points2d = np.dot(proj_mat, points3d_hom.T)
    points2d /= points2d[[2]]
    return points2d[:2].T


def find_inliers(pt, frames):
    inliers = []

    # proj_mats = np.asarray([proj_mat for _, proj_mat in frames])
    # projections = np.asarray([projection for projection, _ in frames]);
    # points2d = proj_hom(pt, proj_mats)
    # points2d_diff = projections - points2d
    # errors = np.linalg.norm(points2d_diff, axis=1)
    # mask = errors < MAX_REPROJ_ERROR

    for projection, proj_mat in frames:
        point2d = proj_hom(pt, proj_mat)
        points2d_diff = projection - point2d
        error = np.linalg.norm(points2d_diff, axis=1)
        if error < MAX_REPROJ_ERROR:
            inliers.append((projection, proj_mat))
    return inliers


def triangulate_multiple_frames_ransac(projections: List[np.ndarray], proj_mats: List[np.ndarray]) -> np.ndarray:
    rnd = random.Random(x=42)

    best_point3d_hom = None
    best_inliers_count = 0
    frames = list(zip(projections, proj_mats))
    for _ in range(10):
        if best_inliers_count == len(frames):
            return best_point3d_hom

        n = rnd.randrange(2, len(frames))
        base = rnd.choices(frames, k=n)
        base_projections, base_proj_mats = zip(*base)

        pt = triangulate_multiple_frames(base_projections, base_proj_mats).reshape(-1, 4)
        inliers = find_inliers(pt, frames)

        if 2 * len(inliers) >= len(frames):
            inliers_projections, inliers_proj_mats = zip(*inliers)
            final_point3d_hom = triangulate_multiple_frames(inliers_projections, inliers_proj_mats).reshape(-1, 4)
            final_inliers = find_inliers(final_point3d_hom, frames)

            if best_point3d_hom is None or len(final_inliers) > best_inliers_count:
                best_point3d_hom = final_point3d_hom
                best_inliers_count = len(final_inliers)
        # else:
        #     print("Failed RANSAC iter")

    # if best_point3d_hom is None:
    #     print("Failed RANSAC")
    # else:
    #     print(best_inliers_count, len(frames))
    return best_point3d_hom


def triangulate_multiple(corner_storage: CornerStorage, view_mats: List[np.ndarray],
                         intrinsic_mat, frames: List[int]):
    proj_mats = [intrinsic_mat @ view_mats[frame] for frame in frames]
    int_ids, int_points = intersect_all(corner_storage, frames)

    points3d_hom = []
    triangulated_ids = []
    failed_ids = []
    for pt_idx, pt_id in enumerate(int_ids):
        print(f"\r{pt_idx}/{len(int_ids)}", end="")
        projections = [points2d[pt_idx] for points2d in int_points]
        point3d_hom = triangulate_multiple_frames_ransac(projections, proj_mats)
        if point3d_hom is not None:
            points3d_hom.append(point3d_hom)
            triangulated_ids.append(pt_id)
        else:
            failed_ids.append(pt_id)
    print("\r", end="")

    points3d = cv2.convertPointsFromHomogeneous(np.asarray(points3d_hom)).reshape(-1, 3)

    # mask = np.ones_like(int_ids).astype(bool)
    # for points2d, proj_mat in zip(int_points, proj_mats):
    #     reproj_errs = compute_reprojection_errors(points3d, points2d, proj_mat)
    #     mask = mask & (reproj_errs < MAX_REPROJ_ERROR)
    # print(f"Remaining: {np.count_nonzero(mask)}/{int_ids.shape[0]}")
    # return int_ids[mask], points3d[mask]

    return np.asarray(triangulated_ids), points3d, np.asarray(failed_ids, dtype=int)


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
                                              intrinsic_mat, None,
                                              iterationsCount=10_000, reprojectionError=MAX_REPROJ_ERROR)
        vm = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
        view_mats.append(vm)

        if frame > 10:
            reference = max(0, frame - 10)
            correspondences = build_correspondences(corner_storage[reference], corners)
            new_cloud, new_triang_id, _ = triangulate_correspondences(correspondences, view_mats[reference], vm,
                                                                      intrinsic_mat, TRIANG_PARAMS)
            point_cloud_builder.add_only_new_points(new_triang_id, new_cloud)

        if frame >= 50 and frame % 5 == 0:
            frames = [frame - 5 * x for x in range(11)]
            aa_ids, aa_pts, failed_ids = triangulate_multiple(corner_storage, view_mats, intrinsic_mat, frames)
            point_cloud_builder.add_points(aa_ids, aa_pts)
            point_cloud_builder.delete_points(failed_ids)

    # view_mats[known_view_1[0]] = vm_1
    # view_mats[known_view_2[0]] = vm_2

    # run bundle adjustment only if we don't have too many points
    # if point_cloud_builder.points.shape[0] < 5000:
    #     run_bundle_adjustment(
    #         intrinsic_mat,
    #         corner_storage,  # noqa: type
    #         MAX_REPROJ_ERROR,
    #         view_mats,
    #         point_cloud_builder,
    #     )

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
