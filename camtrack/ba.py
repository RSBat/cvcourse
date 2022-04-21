import time
from typing import List

import jax
import numpy as np
import jax.numpy as jnp
import optax
import sortednp
import sortednp as snp

from _camtrack import Correspondences
from corners import FrameCorners
from _camtrack import PointCloudBuilder, compute_reprojection_errors, build_correspondences

EPS = 1e-6
lr = 1e-6


def loss_fn(list_of_corners: List[FrameCorners],
            max_inlier_reprojection_error: float,
            proj_mats: List[np.ndarray],
            pc_builder: PointCloudBuilder) -> float:
    res = 0.0
    for corners, proj_mat in zip(list_of_corners, proj_mats):
        correspondences = build_correspondences(pc_builder, corners)
        errors = compute_reprojection_errors(correspondences.points_1, correspondences.points_2, proj_mat)
        res += errors.sum()
    return res / len(proj_mats)


def loss_fn_vm(frame: int,
               proj_mat: np.ndarray,
               list_of_corners: List[FrameCorners],
               max_inlier_reprojection_error: float,
               pc_builder: PointCloudBuilder) -> float:
    corners = list_of_corners[frame]
    correspondences = build_correspondences(pc_builder, corners)
    errors = compute_reprojection_errors(correspondences.points_1, correspondences.points_2, proj_mat)
    return errors.sum() / len(list_of_corners)


# todo: verify optimization
def loss_fn_point(point_id,
                  list_of_corners: List[FrameCorners],
                  max_inlier_reprojection_error: float,
                  proj_mats: List[np.ndarray],
                  pc_builder: PointCloudBuilder) -> float:
    id_arr = np.array([point_id], dtype=pc_builder.ids.dtype)
    _, (_, cloud_idx) = sortednp.intersect(id_arr, pc_builder.ids.flatten(), indices=True)

    res = 0.0
    for corners, proj_mat in zip(list_of_corners[::5], proj_mats[::5]):
        if not sortednp.issubset(id_arr, corners.ids.flatten()):
            continue
        _, (_, corners_idx) = sortednp.intersect(id_arr, corners.ids.flatten(), indices=True)

        errors = compute_reprojection_errors(pc_builder.points[cloud_idx], corners.points[corners_idx], proj_mat)
        res += errors.sum()
    return res / len(proj_mats)


def foo(frame: int,
        intrinsic_mat: np.ndarray,
        list_of_corners: List[FrameCorners],
        max_inlier_reprojection_error: float,
        view_mats: List[np.ndarray],
        pc_builder: PointCloudBuilder,
        loss_scale: float,
        proj_mats: List[np.ndarray]) -> np.ndarray:
    view_mat = view_mats[frame]
    new_view_mat = view_mat.copy()
    it = np.nditer(view_mat, flags=['multi_index'])
    while True:
        og_val = view_mat[it.multi_index]
        view_mat[it.multi_index] = og_val + EPS
        res_add = loss_fn_vm(frame,
                             intrinsic_mat @ view_mat,
                             list_of_corners,
                             max_inlier_reprojection_error,
                             pc_builder)

        view_mat[it.multi_index] = og_val - EPS
        res_sub = loss_fn_vm(frame,
                             intrinsic_mat @ view_mat,
                             list_of_corners,
                             max_inlier_reprojection_error,
                             pc_builder)

        view_mat[it.multi_index] = og_val

        grad = loss_scale * (res_add - res_sub) / (2 * EPS)
        # print("vm", og_val, grad)
        new_view_mat[it.multi_index] = og_val - lr * grad

        if not it.iternext():
            break
    return new_view_mat


def create_loss_fn_jax(intrinsic_mat: np.ndarray,
                       list_of_corners: List[FrameCorners],
                       max_inlier_reprojection_error: float,
                       # proj_mats: np.ndarray,
                       ids: np.ndarray):
    def f(points, view_mats):
        res = 0.0
        for frame, corners in enumerate(list_of_corners):
            proj_mat = jnp.dot(intrinsic_mat, view_mats[frame])
            ids_1 = ids.flatten()
            ids_2 = corners.ids.flatten()
            _, (indices_1, indices_2) = snp.intersect(ids_1, ids_2, indices=True)
            correspondences = Correspondences(
                ids_1[indices_1],
                points[indices_1],
                corners.points[indices_2]
            )

            points3d = jnp.pad(correspondences.points_1, ((0, 0), (0, 1)), 'constant', constant_values=(1,))
            points2d = jnp.dot(proj_mat, points3d.T)
            points2d /= points2d[jnp.array([2])]
            projected_points = points2d[:2].T

            points2d_diff = correspondences.points_2 - projected_points
            errors = jnp.linalg.norm(points2d_diff, axis=1)
            res += errors.sum()
        return res / len(list_of_corners)
    return f


def bar(intrinsic_mat: np.ndarray,
        list_of_corners: List[FrameCorners],
        max_inlier_reprojection_error: float,
        view_mats: List[np.ndarray],
        pc_builder: PointCloudBuilder,
        loss_scale: float,
        proj_mats: List[np.ndarray]) -> np.ndarray:
    diff_st = time.time()
    new_points = pc_builder.points.copy()
    it = np.nditer(pc_builder.points, flags=['multi_index'])
    while True:
        # print(f"\r{it.multi_index}/{pc_builder.points.shape}", end="")

        og_val = pc_builder.points[it.multi_index]
        pc_builder.points[it.multi_index] = og_val + EPS
        res_add = loss_fn_point(
            pc_builder.ids[it.multi_index[0]][0],
            list_of_corners,
            max_inlier_reprojection_error,
            proj_mats,
            pc_builder
        )
        # res_add = loss_fn(
        #     list_of_corners,
        #     max_inlier_reprojection_error,
        #     proj_mats,
        #     pc_builder
        # )

        pc_builder.points[it.multi_index] = og_val - EPS
        res_sub = loss_fn_point(
            pc_builder.ids[it.multi_index[0]][0],
            list_of_corners,
            max_inlier_reprojection_error,
            proj_mats,
            pc_builder
        )
        # res_sub = loss_fn(
        #     list_of_corners,
        #     max_inlier_reprojection_error,
        #     proj_mats,
        #     pc_builder
        # )

        pc_builder.points[it.multi_index] = og_val

        grad = loss_scale * (res_add - res_sub) / (2 * EPS)
        # print("pt", og_val, grad, loss_scale * jax_grad[it.multi_index])
        new_points[it.multi_index] = og_val - lr * grad

        if not it.iternext():
            break
    diff_end = time.time()
    # print("Jax", jax_end - jax_st)
    # print("Diff", diff_end - diff_st)
    return new_points


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:
    # TODO: implement
    # You may modify pc_builder using 'update_points' method

    proj_mats = [intrinsic_mat @ view_mat for view_mat in view_mats]

    initial_loss = loss_fn(list_of_corners, max_inlier_reprojection_error, proj_mats, pc_builder)
    loss_scale = 100 / initial_loss
    print(initial_loss)

    jax_pre_st = time.time()
    jpt = jnp.copy(pc_builder.points)
    jvm = jnp.asarray(view_mats)

    loss_fn_jax = create_loss_fn_jax(intrinsic_mat, list_of_corners, max_inlier_reprojection_error,
                                     pc_builder.ids)
    loss_fn_grad = jax.grad(loss_fn_jax, argnums=(0, 1))
    jax_pre_end = time.time()
    print("Jax pre", jax_pre_end - jax_pre_st)

    for _ in range(1):
        jax_st = time.time()
        for jax_step in range(10):
            jpt_grad, jvm_grad = loss_fn_grad(jpt, jvm)
            jpt = jpt - lr * loss_scale * jpt_grad
            jvm = jvm - lr * loss_scale * jvm_grad
        jax_end = time.time()
        print("Jax final loss: ", loss_fn_jax(jpt, jvm))

        reg_st = time.time()
        proj_mats = [intrinsic_mat @ view_mat for view_mat in view_mats]

        new_view_mats = []
        for frame in range(len(view_mats)):
            print(f"\rframe: {frame}/{len(view_mats)}", end="")
            new_view_mat = foo(frame,
                               intrinsic_mat,
                               list_of_corners,
                               max_inlier_reprojection_error,
                               view_mats,
                               pc_builder,
                               loss_scale,
                               proj_mats)
            new_view_mats.append(new_view_mat)

        print("\rpoints", end="")
        new_points = bar(intrinsic_mat,
                         list_of_corners,
                         max_inlier_reprojection_error,
                         view_mats,
                         pc_builder,
                         loss_scale,
                         proj_mats)

        view_mats = new_view_mats
        pc_builder.update_points(pc_builder.ids, new_points)

        print("\r", end="")
        print(loss_fn(list_of_corners,
                      max_inlier_reprojection_error,
                      proj_mats,
                      pc_builder))
        reg_end = time.time()
        print("Jax", jax_end - jax_st)
        print("Regq", reg_end - reg_st)

    return view_mats
