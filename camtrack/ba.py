import time
from typing import List

import jax
import numpy as np
import jax.numpy as jnp
import optax
import sortednp as snp

from _camtrack import Correspondences, PointCloudBuilder
from corners import FrameCorners

lr = 1e-5


def create_loss_fn_jax(intrinsic_mat: np.ndarray,
                       list_of_corners: List[FrameCorners],
                       max_inlier_reprojection_error: float,
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
            errors_mask = errors < max_inlier_reprojection_error

            res += (points2d_diff[errors_mask] ** 2).sum()
        return res / len(list_of_corners)
    return f


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:
    loss_fn_jax = create_loss_fn_jax(intrinsic_mat, list_of_corners, max_inlier_reprojection_error,
                                     pc_builder.ids)
    jpt = jnp.copy(pc_builder.points)
    jvm = jnp.asarray(view_mats)

    initial_loss = loss_fn_jax(jpt, jvm)
    loss_scale = 100 / initial_loss
    print("Initial loss:", initial_loss)

    loss_fn_grad = jax.grad(loss_fn_jax, argnums=(0, 1))
    pt_adam = optax.adam(lr)
    pt_state = pt_adam.init(jpt)
    vm_adam = optax.adam(lr)
    vm_state = vm_adam.init(jvm)

    for _ in range(10):
        jpt_grad, jvm_grad = loss_fn_grad(jpt, jvm)

        pt_updates, pt_state = pt_adam.update(loss_scale * jpt_grad, pt_state)
        jpt = optax.apply_updates(jpt, pt_updates)

        vm_updates, vm_state = pt_adam.update(loss_scale * jvm_grad, vm_state)
        jvm = optax.apply_updates(jvm, vm_updates)

        print("\rLoss:", loss_fn_jax(jpt, jvm), end="")
    print()

    view_mats = [np.asarray(jvm[frame]) for frame in range(len(view_mats))]
    pc_builder.update_points(pc_builder.ids, np.asarray(jpt))

    return view_mats
