from typing import List

import numpy as np
import sortednp

from corners import FrameCorners
from _camtrack import PointCloudBuilder, compute_reprojection_errors, build_correspondences

EPS = 1e-6
lr = 1e-6


def loss_fn(intrinsic_mat: np.ndarray,
            list_of_corners: List[FrameCorners],
            max_inlier_reprojection_error: float,
            view_mats: List[np.ndarray],
            pc_builder: PointCloudBuilder) -> float:
    res = 0.0
    for corners, view_mat in zip(list_of_corners, view_mats):
        correspondences = build_correspondences(pc_builder, corners)
        proj_mat = intrinsic_mat @ view_mat
        errors = compute_reprojection_errors(correspondences.points_1, correspondences.points_2, proj_mat)
        res += errors.sum()
    return res / len(view_mats)


# todo: verify optimization
def loss_fn_point(point_id,
                  intrinsic_mat: np.ndarray,
                  list_of_corners: List[FrameCorners],
                  max_inlier_reprojection_error: float,
                  view_mats: List[np.ndarray],
                  pc_builder: PointCloudBuilder):
    _, (_, cloud_idx) = sortednp.intersect(np.asarray([point_id]), pc_builder.ids.flatten(), indices=True)

    res = 0.0
    for corners, view_mat in zip(list_of_corners, view_mats):
        if not sortednp.isitem(point_id, corners.ids.flatten()):
            continue
        _, (_, corners_idx) = sortednp.intersect(np.asarray([point_id]), corners.ids.flatten(), indices=True)

        proj_mat = intrinsic_mat @ view_mat
        errors = compute_reprojection_errors(pc_builder.points[cloud_idx], corners.points[corners_idx], proj_mat)
        res += errors.sum()
    return res / len(view_mats)


def foo(frame: int,
        intrinsic_mat: np.ndarray,
        list_of_corners: List[FrameCorners],
        max_inlier_reprojection_error: float,
        view_mats: List[np.ndarray],
        pc_builder: PointCloudBuilder,
        loss_scale: float) -> np.ndarray:
    view_mat = view_mats[frame]
    new_view_mat = view_mat.copy()
    it = np.nditer(view_mat, flags=['multi_index'])
    while True:
        og_val = view_mat[it.multi_index]
        view_mat[it.multi_index] = og_val + EPS
        res_add = loss_fn(intrinsic_mat,
                          list_of_corners,
                          max_inlier_reprojection_error,
                          view_mats,
                          pc_builder)

        view_mat[it.multi_index] = og_val - EPS
        res_sub = loss_fn(intrinsic_mat,
                          list_of_corners,
                          max_inlier_reprojection_error,
                          view_mats,
                          pc_builder)

        view_mat[it.multi_index] = og_val

        grad = loss_scale * (res_add - res_sub) / (2 * EPS)
        # print("vm", og_val, grad)
        new_view_mat[it.multi_index] = og_val - lr * grad

        if not it.iternext():
            break
    return new_view_mat


def bar(intrinsic_mat: np.ndarray,
        list_of_corners: List[FrameCorners],
        max_inlier_reprojection_error: float,
        view_mats: List[np.ndarray],
        pc_builder: PointCloudBuilder,
        loss_scale: float) -> np.ndarray:
    new_points = pc_builder.points.copy()
    it = np.nditer(pc_builder.points, flags=['multi_index'])
    while True:
        print(f"\r{it.multi_index}/{pc_builder.points.shape}", end="")

        og_val = pc_builder.points[it.multi_index]
        pc_builder.points[it.multi_index] = og_val + EPS
        res_add = loss_fn_point(
            pc_builder.ids[it.multi_index[0]][0],
            intrinsic_mat,
            list_of_corners,
            max_inlier_reprojection_error,
            view_mats,
            pc_builder
        )

        pc_builder.points[it.multi_index] = og_val - EPS
        res_sub = loss_fn_point(
            pc_builder.ids[it.multi_index[0]][0],
            intrinsic_mat,
            list_of_corners,
            max_inlier_reprojection_error,
            view_mats,
            pc_builder
        )

        pc_builder.points[it.multi_index] = og_val

        grad = loss_scale * (res_add - res_sub) / (2 * EPS)
        # print("pt", og_val, grad)
        new_points[it.multi_index] = og_val - lr * grad

        if not it.iternext():
            break
    return new_points


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:
    # TODO: implement
    # You may modify pc_builder using 'update_points' method

    initial_loss = loss_fn(intrinsic_mat, list_of_corners, max_inlier_reprojection_error, view_mats, pc_builder)
    loss_scale = 100 / initial_loss
    print(initial_loss)

    for _ in range(1):
        new_view_mats = []
        for frame in range(len(view_mats)):
            print(f"\rframe: {frame}/{len(view_mats)}", end="")
            new_view_mat = foo(frame,
                               intrinsic_mat,
                               list_of_corners,
                               max_inlier_reprojection_error,
                               view_mats,
                               pc_builder,
                               loss_scale)
            new_view_mats.append(new_view_mat)

        print("\rpoints", end="")
        new_points = bar(intrinsic_mat,
                         list_of_corners,
                         max_inlier_reprojection_error,
                         view_mats,
                         pc_builder,
                         loss_scale)

        view_mats = new_view_mats
        pc_builder.update_points(pc_builder.ids, new_points)

        print("\r", end="")
        print(loss_fn(intrinsic_mat,
                      list_of_corners,
                      max_inlier_reprojection_error,
                      view_mats,
                      pc_builder))

    return view_mats
