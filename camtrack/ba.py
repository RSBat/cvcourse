import time
from typing import List

import cv2
import jax
import numpy as np
import jax.numpy as jnp
import optax
import sortednp as snp

from _camtrack import Correspondences, PointCloudBuilder, view_mat3x4_to_pose, pose_to_view_mat3x4
from data3d import Pose
from corners import FrameCorners

lr = 1e-5


def jax_Rodrigues(rvec):
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


def to_view_mat(rvecs, tvecs, frame):
    rmat = jax_Rodrigues(rvecs[frame])
    tvec = tvecs[frame]
    pose = Pose(rmat, tvec)
    return jnp.hstack((
        pose.r_mat.T,
        pose.r_mat.T @ -pose.t_vec.reshape(-1, 1)
    ))


def create_loss_fn_jax(intrinsic_mat: np.ndarray,
                       list_of_corners: List[FrameCorners],
                       max_inlier_reprojection_error: float,
                       ids: np.ndarray):
    def f(points, rvecs, tvecs):
        res = 0.0
        for frame, corners in enumerate(list_of_corners):
            view_mat = to_view_mat(rvecs, tvecs, frame)
            proj_mat = jnp.dot(intrinsic_mat, view_mat)

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

    poses = [view_mat3x4_to_pose(view_mat) for view_mat in view_mats]
    rvecs = [cv2.Rodrigues(pose.r_mat)[0] for pose in poses]
    tvecs = [pose.t_vec for pose in poses]
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
