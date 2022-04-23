import time
from typing import List

import cv2
import jax
import numpy as np
import jax.numpy as jnp
import optax
import sortednp as snp

from _camtrack import PointCloudBuilder
from corners import FrameCorners

lr = 1e-5


def jax_Rodrigues(rvec: jnp.ndarray) -> jnp.ndarray:
    theta = jnp.linalg.norm(rvec)
    if theta < 1e-6:
        rotation_mat = jnp.eye(3, dtype=float)
    else:
        r = rvec / theta
        I = jnp.eye(3, dtype=float)
        r_rT = r @ r.T
        r_cross = jnp.cross(r.T, -jnp.identity(3))

        rotation_mat = jnp.cos(theta) * I + (1 - jnp.cos(theta)) * r_rT + jnp.sin(theta) * r_cross
    return rotation_mat


def to_view_mat(rvecs: jnp.ndarray, tvecs: jnp.ndarray, frame: int) -> jnp.ndarray:
    rmat = jax_Rodrigues(rvecs[frame])
    tvec = tvecs[frame]
    return jnp.hstack((
        rmat,
        tvec.reshape(-1, 1)
    ))


def create_loss_fn_jax(intrinsic_mat: np.ndarray,
                       list_of_corners: List[FrameCorners],
                       max_inlier_reprojection_error: float,
                       ids: np.ndarray):
    masks = []
    matched_corners = []
    for frame_corners in list_of_corners:
        ids_1 = ids.flatten()
        ids_2 = frame_corners.ids.flatten()
        _, (indices_1, indices_2) = snp.intersect(ids_1, ids_2, indices=True)
        masks.append(indices_1)
        matched_corners.append(frame_corners.points[indices_2])

    def f(points, rvecs, tvecs):
        res = 0.0
        points3d = jnp.pad(points, ((0, 0), (0, 1)), 'constant', constant_values=(1,))

        for frame, (corners, indices) in enumerate(zip(matched_corners, masks)):
            view_mat = to_view_mat(rvecs, tvecs, frame)
            proj_mat = jnp.dot(intrinsic_mat, view_mat)

            points2d = jnp.dot(proj_mat, points3d[indices].T)
            points2d /= points2d[jnp.array([2])]
            projected_points = points2d[:2].T

            points2d_diff = corners - projected_points
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

    rvecs = []
    tvecs = []
    for view_mat in view_mats:
        r_mat = view_mat[:, :3]
        t_vec = view_mat[:, 3]
        rvecs.append(cv2.Rodrigues(r_mat)[0])
        tvecs.append(t_vec)

    jrv = jnp.asarray(rvecs)
    jtv = jnp.asarray(tvecs)

    initial_loss = loss_fn_jax(jpt, jrv, jtv)
    loss_scale = 100 / initial_loss
    print("Initial loss:", initial_loss)

    loss_fn_grad = jax.grad(loss_fn_jax, argnums=(0, 1, 2))
    pt_adam = optax.adam(lr)
    pt_state = pt_adam.init(jpt)
    rv_adam = optax.adam(lr)
    rv_state = rv_adam.init(jrv)
    tv_adam = optax.adam(lr)
    tv_state = tv_adam.init(jtv)

    for _ in range(10):
        jpt_grad, jrv_grad, jtv_grad = loss_fn_grad(jpt, jrv, jtv)

        pt_updates, pt_state = pt_adam.update(loss_scale * jpt_grad, pt_state)
        jpt = optax.apply_updates(jpt, pt_updates)

        rv_updates, rv_state = pt_adam.update(loss_scale * jrv_grad, rv_state)
        jrv = optax.apply_updates(jrv, rv_updates)

        tv_updates, tv_state = pt_adam.update(loss_scale * jtv_grad, tv_state)
        jtv = optax.apply_updates(jtv, tv_updates)

        print("\rLoss:", loss_fn_jax(jpt, jrv, jtv), end="")
    print()

    view_mats = [to_view_mat(jrv, jtv, frame) for frame in range(len(view_mats))]
    pc_builder.update_points(pc_builder.ids, np.asarray(jpt))

    return view_mats
