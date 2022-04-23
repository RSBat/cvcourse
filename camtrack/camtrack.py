#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import itertools
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp

from ba import run_bundle_adjustment
from _corners import FrameCorners
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
    view_mat3x4_to_pose, rodrigues_and_translation_to_view_mat3x4, eye3x4,
)

MAX_REPROJ_ERROR = 5.0
TRIANG_PARAMS = TriangulationParameters(max_reprojection_error=MAX_REPROJ_ERROR,
                                        min_triangulation_angle_deg=2,
                                        min_depth=0)


def recover_pose(corners_1, corners_2, intrinsic_mat) -> Tuple[Optional[Pose], float, float]:
    correspondences = build_correspondences(corners_1, corners_2)
    E, mask = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2,
                                   intrinsic_mat, method=cv2.RANSAC)
    pose_inliers, R, t, pose_mask = cv2.recoverPose(E, correspondences.points_1, correspondences.points_2,
                                                    intrinsic_mat, mask=mask)

    vm_1 = eye3x4()
    vm_2 = np.hstack([R, t])

    H, homography_mask = cv2.findHomography(correspondences.points_1, correspondences.points_2,
                                            method=cv2.RANSAC, ransacReprojThreshold=MAX_REPROJ_ERROR)

    _, triang_ids, median_cos = triangulate_correspondences(correspondences, vm_1, vm_2,
                                                            intrinsic_mat, TRIANG_PARAMS)

    if triang_ids.shape[0] < 20:
        return view_mat3x4_to_pose(vm_2), 1, 1

    ratio = np.count_nonzero(homography_mask) / pose_inliers if pose_inliers != 0 else 1
    return view_mat3x4_to_pose(vm_2), ratio, median_cos


def init_views(corner_storage, intrinsic_mat):
    eps = 1e-2

    best_result = None
    best_ratio = 1.
    best_cos = 1.
    for offset in [50, 30, 20, 10]:
        for frame_1 in range(len(corner_storage) - offset):
            frame_2 = frame_1 + offset
            pose_2, ratio, median_cos = recover_pose(corner_storage[frame_1], corner_storage[frame_2], intrinsic_mat)
            print(f"\rFrames: {frame_1} {frame_2}, quality: {1 - ratio}, cos: {median_cos}", end="")

            better_ratio = ratio + eps < best_ratio
            better_angle = np.isclose(ratio, best_ratio, eps) and median_cos + eps < best_cos
            if best_result is None or better_ratio or better_angle:
                pose_1 = view_mat3x4_to_pose(eye3x4())
                known_view_1 = frame_1, pose_1
                known_view_2 = frame_2, pose_2

                best_result = known_view_1, known_view_2
                best_ratio = ratio
                best_cos = median_cos

    print(f"\rSelected frames: {best_result[0][0]} {best_result[1][0]}, quality: {1 - best_ratio}, cos: {best_cos}", end="")
    print()
    return best_result


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


def find_inliers(pt, frames):
    inliers = []

    proj_mats = np.asarray([proj_mat for _, proj_mat in frames])
    projections = np.asarray([projection for projection, _ in frames])
    points2d_hom = (proj_mats @ pt.T).reshape(-1, 3)
    points2d = (points2d_hom / points2d_hom[:, 2].reshape(-1, 1))[:, :2]
    points2d_diff = projections - points2d
    errors = np.linalg.norm(points2d_diff, axis=1)
    mask = errors < MAX_REPROJ_ERROR

    for good, frame in zip(mask, frames):
        if good:
            inliers.append(frame)
    return inliers


def triangulate_multiple_frames_ransac(projections: List[np.ndarray], proj_mats: List[np.ndarray]) -> np.ndarray:
    rnd = random.Random(x=42)

    best_point3d_hom = None
    best_inliers_count = 0
    frames = list(zip(projections, proj_mats))
    frames_count = len(frames)
    for _ in range(10):
        if best_inliers_count == frames_count:
            return best_point3d_hom

        n = rnd.randrange(2, frames_count)
        base = rnd.choices(frames, k=n)
        base_projections, base_proj_mats = zip(*base)

        pt = triangulate_multiple_frames(base_projections, base_proj_mats).reshape(-1, 4)
        inliers = find_inliers(pt, frames)

        successful = 2 * len(inliers) >= frames_count
        if successful and (best_point3d_hom is None or len(inliers) > best_inliers_count):
            best_point3d_hom = pt
            best_inliers_count = len(inliers)

        # almost does not affect quality
        # if 2 * len(inliers) >= frames_count:
        #     inliers_projections, inliers_proj_mats = zip(*inliers)
        #     final_point3d_hom = triangulate_multiple_frames(inliers_projections, inliers_proj_mats).reshape(-1, 4)
        #     final_inliers = find_inliers(final_point3d_hom, frames)
        #
        #     if best_point3d_hom is None or len(final_inliers) > best_inliers_count:
        #         best_point3d_hom = final_point3d_hom
        #         best_inliers_count = len(final_inliers)

    return best_point3d_hom


def triangulate_multiple(corner_storage: CornerStorage, proj_mats: List[np.ndarray],
                         point_ids: np.ndarray, step: int = 5):
    points3d_hom = []
    triangulated_ids = []
    failed_ids = []
    corner_ids = [corners.ids.flatten() for corners in corner_storage]
    for pt_id in point_ids:
        projections = []
        selected_proj_mats = []
        for corners, ids, proj_mat in zip(corner_storage[::step], corner_ids[::step], proj_mats[::step]):
            if proj_mat is None:
                continue
            _, (idx, _) = snp.intersect(ids, pt_id, indices=True)
            if idx.shape[0] == 0:
                continue
            projections.append(corners.points[idx][0])
            selected_proj_mats.append(proj_mat)

        if len(projections) < 5:
            continue

        while len(projections) > 20:
            projections = projections[::2]
            selected_proj_mats = selected_proj_mats[::2]

        point3d_hom = triangulate_multiple_frames_ransac(projections, selected_proj_mats)
        if point3d_hom is not None:
            points3d_hom.append(point3d_hom)
            triangulated_ids.append(pt_id)
        else:
            failed_ids.append(pt_id)

    if len(points3d_hom) > 0:
        points3d = cv2.convertPointsFromHomogeneous(np.asarray(points3d_hom)).reshape(-1, 3)
    else:
        points3d = np.asarray([], dtype=float)
    return np.asarray(triangulated_ids, dtype=int), points3d, np.asarray(failed_ids, dtype=int)


def init_cloud(corner_storage: CornerStorage,
               known_view_1: Tuple[int, Pose],
               known_view_2: Tuple[int, Pose],
               intrinsic_mat: np.ndarray) -> PointCloudBuilder:
    vm_1 = pose_to_view_mat3x4(known_view_1[1])
    vm_2 = pose_to_view_mat3x4(known_view_2[1])

    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    cloud, id_triangulated, _ = triangulate_correspondences(correspondences, vm_1, vm_2, intrinsic_mat, TRIANG_PARAMS)
    return PointCloudBuilder(id_triangulated, cloud)


def find_view_mat(pc_builder: PointCloudBuilder,
                  corners: FrameCorners,
                  intrinsic_mat: np.ndarray) -> np.ndarray:
    _, (lhs, rhs) = snp.intersect(pc_builder.ids.flatten(), corners.ids.flatten(), indices=True)
    _, rvec, tvec, _ = cv2.solvePnPRansac(
        pc_builder.points[lhs].copy(), corners.points[rhs].copy(),
        intrinsic_mat, None,
        iterationsCount=10_000, reprojectionError=MAX_REPROJ_ERROR,
    )
    view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
    return view_mat


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
    frame_count = len(corner_storage)

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = init_views(corner_storage, intrinsic_mat)

    point_cloud_builder = init_cloud(corner_storage, known_view_1, known_view_2, intrinsic_mat)

    view_mats: List[Optional[np.ndarray]] = [None] * frame_count
    proj_mats: List[Optional[np.ndarray]] = [None] * frame_count
    frame_iter = itertools.chain(range(known_view_1[0], frame_count),
                                 range(known_view_1[0] - 1, -1, -1))
    for n, frame in enumerate(frame_iter):
        print(f"\rProcessing frame {frame + 1} ({n + 1}/{frame_count})", end="")
        corners = corner_storage[frame]
        view_mat = find_view_mat(point_cloud_builder, corners, intrinsic_mat)
        view_mats[frame] = view_mat
        proj_mats[frame] = intrinsic_mat @ view_mat

        if frame % 5 == 0:
            reference = max(0, frame - 10)
            if view_mats[reference] is None:
                reference = min(frame + 10, frame_count - 1)

            if view_mats[reference] is not None:
                correspondences = build_correspondences(corner_storage[reference], corners)
                new_cloud, new_triang_id, _ = triangulate_correspondences(
                    correspondences, view_mats[reference], view_mat,
                    intrinsic_mat, TRIANG_PARAMS
                )
                point_cloud_builder.add_only_new_points(new_triang_id, new_cloud)

        if frame % 10 == 0:
            multi_ids, multi_pts, failed_ids = triangulate_multiple(corner_storage, proj_mats, corners.ids)
            point_cloud_builder.add_points(multi_ids, multi_pts)
            point_cloud_builder.delete_points(failed_ids)
    print()

    multi_ids, multi_pts, failed_ids = triangulate_multiple(corner_storage, proj_mats,
                                                            point_cloud_builder.ids, step=2)
    point_cloud_builder.add_points(multi_ids, multi_pts)
    point_cloud_builder.delete_points(failed_ids)

    # run bundle adjustment only if we don't have too many points
    if frame_count < 100 and point_cloud_builder.points.shape[0] < 5000:
        view_mats = run_bundle_adjustment(
            intrinsic_mat,
            corner_storage,  # noqa: type
            MAX_REPROJ_ERROR,
            view_mats,
            point_cloud_builder,
        )

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
