#!/usr/bin/env python3
"""Core rigid registration methods used in CMF-PAIR.

Task: estimate a rigid transform T such that T @ source ≈ target, where
source is an IOS / DDC tooth mesh and target is the CT tooth mesh.

Default geometric protocol (paper):
  voxel downsample 0.8 mm
  coarse ICP 5.0 mm → fine ICP 1.0 mm
  unified Open3D evaluate_registration at delta = 1.0 mm
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

METHODS = ("single_pca_icp", "pca_icp", "fpfh_ransac_icp")
PCA_FLIPS = (
    ("pca-000", np.diag([1.0, 1.0, 1.0])),
    ("pca-011", np.diag([1.0, -1.0, -1.0])),
    ("pca-101", np.diag([-1.0, 1.0, -1.0])),
    ("pca-110", np.diag([-1.0, -1.0, 1.0])),
)


def reg_module():
    import open3d as o3d

    if hasattr(o3d, "pipelines") and hasattr(o3d.pipelines, "registration"):
        return o3d.pipelines.registration
    return o3d.registration


def mesh_to_pcd(mesh):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    return pcd


def estimate_normals(pcd, voxel: float) -> None:
    import open3d as o3d

    radius = max(float(voxel) * 2.0, 1e-6)
    try:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    except TypeError:
        pcd.estimate_normals()


def downsample_for_registration(pcd, voxel: float):
    out = pcd if voxel <= 0 else pcd.voxel_down_sample(float(voxel))
    estimate_normals(out, voxel if voxel > 0 else 1.0)
    return out


def points(pcd) -> np.ndarray:
    return np.asarray(pcd.points, dtype=np.float64)


def pca_frame(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = pts.mean(axis=0)
    x = pts - center
    cov = (x.T @ x) / max(len(x) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    frame = vecs[:, np.argsort(vals)[::-1]]
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1.0
    return center, frame


def make_transform(rotation: np.ndarray, src_center: np.ndarray, tgt_center: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = tgt_center - rotation @ src_center
    return T


def pca_initial_transforms(src_pts: np.ndarray, tgt_pts: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Proper-rotation PCA alignments (determinant +1 only)."""
    src_center, src_frame = pca_frame(src_pts)
    tgt_center, tgt_frame = pca_frame(tgt_pts)
    out = []
    for label, sign in PCA_FLIPS:
        R = tgt_frame @ sign @ src_frame.T
        if np.linalg.det(R) < 0:
            continue
        out.append((label, make_transform(R, src_center, tgt_center)))
    return out


def icp(
    src_pcd,
    tgt_pcd,
    init: np.ndarray,
    coarse_dist: float,
    fine_dist: float,
    *,
    point_to_plane: bool = True,
):
    reg = reg_module()
    if point_to_plane:
        try:
            estimation = reg.TransformationEstimationPointToPlane()
        except Exception:
            estimation = reg.TransformationEstimationPointToPoint()
    else:
        estimation = reg.TransformationEstimationPointToPoint()

    coarse = reg.registration_icp(
        source=src_pcd,
        target=tgt_pcd,
        max_correspondence_distance=float(coarse_dist),
        init=init,
        estimation_method=estimation,
        criteria=reg.ICPConvergenceCriteria(max_iteration=1500),
    )
    fine = reg.registration_icp(
        source=src_pcd,
        target=tgt_pcd,
        max_correspondence_distance=float(fine_dist),
        init=coarse.transformation,
        estimation_method=estimation,
        criteria=reg.ICPConvergenceCriteria(max_iteration=2500),
    )
    return np.asarray(fine.transformation, dtype=np.float64), float(fine.fitness), float(fine.inlier_rmse)


def evaluate_full(src_mesh, tgt_mesh, T: np.ndarray, delta: float):
    reg = reg_module()
    return reg.evaluate_registration(mesh_to_pcd(src_mesh), mesh_to_pcd(tgt_mesh), float(delta), T)


def preprocess_fpfh(pcd, voxel: float):
    import open3d as o3d

    reg = reg_module()
    down = downsample_for_registration(pcd, voxel)
    radius_feature = max(voxel * 5.0, 1e-6)
    try:
        fpfh = reg.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
    except TypeError:
        fpfh = reg.compute_fpfh_feature(down)
    return down, fpfh


def ransac_global(src_down, tgt_down, src_fpfh, tgt_fpfh, distance: float) -> np.ndarray:
    reg = reg_module()
    estimation = reg.TransformationEstimationPointToPoint(False)
    try:
        result = reg.registration_ransac_based_on_feature_matching(
            src_down,
            tgt_down,
            src_fpfh,
            tgt_fpfh,
            False,
            float(distance),
            estimation,
            4,
            [
                reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                reg.CorrespondenceCheckerBasedOnDistance(float(distance)),
            ],
            reg.RANSACConvergenceCriteria(400000, 500),
        )
    except TypeError:
        result = reg.registration_ransac_based_on_feature_matching(
            src_down,
            tgt_down,
            src_fpfh,
            tgt_fpfh,
            float(distance),
            estimation,
            4,
            [],
            reg.RANSACConvergenceCriteria(8000000, 1000),
        )
    return np.asarray(result.transformation, dtype=np.float64)


def run_pca_icp(src_pcd_reg, tgt_pcd_reg, coarse_dist: float, fine_dist: float):
    """Multi-hypothesis PCA-ICP: try 4 proper PCA sign flips, keep best fitness/RMSE."""
    best = None
    for label, init in pca_initial_transforms(points(src_pcd_reg), points(tgt_pcd_reg)):
        try:
            T, fit, rmse = icp(src_pcd_reg, tgt_pcd_reg, init, coarse_dist, fine_dist)
        except Exception as exc:
            if best is None:
                best = (label, np.eye(4), -1.0, float("inf"), f"failed: {exc}")
            continue
        if best is None or (-fit, rmse) < (-best[2], best[3]):
            best = (label, T, fit, rmse, "ok")
    if best is None:
        return "none", np.eye(4), -1.0, float("inf"), "no PCA candidate"
    return best


def run_single_pca_icp(src_pcd_reg, tgt_pcd_reg, coarse_dist: float, fine_dist: float):
    """Single-hypothesis PCA-ICP: only the first proper PCA sign (pca-000)."""
    candidates = pca_initial_transforms(points(src_pcd_reg), points(tgt_pcd_reg))
    if not candidates:
        return "none", np.eye(4), -1.0, float("inf"), "no PCA candidate"
    label, init = candidates[0]
    try:
        T, fit, rmse = icp(src_pcd_reg, tgt_pcd_reg, init, coarse_dist, fine_dist)
        return f"single-{label}", T, fit, rmse, "ok"
    except Exception as exc:
        return f"single-{label}", np.eye(4), -1.0, float("inf"), f"failed: {exc}"


def run_fpfh_ransac_icp(src_pcd_full, tgt_pcd_full, voxel: float, coarse_dist: float, fine_dist: float):
    """FPFH + RANSAC global alignment, then point-to-point ICP."""
    src_down, src_fpfh = preprocess_fpfh(src_pcd_full, voxel)
    tgt_down, tgt_fpfh = preprocess_fpfh(tgt_pcd_full, voxel)
    init = ransac_global(src_down, tgt_down, src_fpfh, tgt_fpfh, max(voxel * 3.0, fine_dist * 2.0))
    T, fit, rmse = icp(src_down, tgt_down, init, coarse_dist, fine_dist, point_to_plane=False)
    return "fpfh-ransac", T, fit, rmse, "ok"


def register_meshes(
    src_mesh,
    tgt_mesh,
    method: str = "pca_icp",
    *,
    voxel_mm: float = 0.8,
    coarse_mm: float = 5.0,
    fine_mm: float = 1.0,
    delta: float = 1.0,
) -> dict[str, Any]:
    """Register two Open3D triangle meshes. Returns transform and unified metrics."""
    if method not in METHODS:
        raise ValueError(f"Unsupported method {method!r}. Choose from {METHODS}.")

    src_full = mesh_to_pcd(src_mesh)
    tgt_full = mesh_to_pcd(tgt_mesh)
    src_reg = downsample_for_registration(src_full, voxel_mm)
    tgt_reg = downsample_for_registration(tgt_full, voxel_mm)

    if method == "single_pca_icp":
        label, T, reg_fit, reg_rmse, message = run_single_pca_icp(
            src_reg, tgt_reg, coarse_mm, fine_mm
        )
    elif method == "pca_icp":
        label, T, reg_fit, reg_rmse, message = run_pca_icp(src_reg, tgt_reg, coarse_mm, fine_mm)
    else:
        label, T, reg_fit, reg_rmse, message = run_fpfh_ransac_icp(
            src_full, tgt_full, voxel_mm, coarse_mm, fine_mm
        )

    ev = evaluate_full(src_mesh, tgt_mesh, T, delta)
    return {
        "method": method,
        "selected_init": label,
        "transform": np.asarray(T, dtype=np.float64),
        "registration_fitness_downsample": float(reg_fit),
        "registration_rmse_downsample": float(reg_rmse) if math.isfinite(reg_rmse) else float("inf"),
        "fitness": float(ev.fitness),
        "inlier_rmse": float(ev.inlier_rmse),
        "message": message,
    }
