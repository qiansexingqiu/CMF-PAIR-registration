"""Dataset folder conventions used by CMF-PAIR registration scripts."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

RESULT_SUBDIR = {
    "single_pca_icp": "traditional_single_pca_icp_result",
    "pca_icp": "traditional_pca_icp_result",
    "fpfh_ransac_icp": "traditional_fpfh_ransac_icp_result",
}
CSV_NAME = {
    "single_pca_icp": "unified_fitness_single_pca_icp_delta{delta_tag}.csv",
    "pca_icp": "unified_fitness_pca_icp_delta{delta_tag}.csv",
    "fpfh_ransac_icp": "unified_fitness_fpfh_ransac_icp_delta{delta_tag}.csv",
}


def patient_sort_key(path: Path) -> int:
    m = re.search(r"case(\d+)$", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", path.name)
    return int(m.group(1)) if m else 999999


def iter_patients(root: Path, only: set[str] | None = None) -> list[Path]:
    patients = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("Patient")],
        key=patient_sort_key,
    )
    if only:
        patients = [p for p in patients if p.name in only]
    return patients


def delta_tag(delta: float) -> str:
    if abs(delta - round(delta)) < 1e-9:
        return str(int(round(delta)))
    return str(delta).replace(".", "p")


def load_pair(patient_dir: Path, pid: str, jaw: str):
    import open3d as o3d

    jaw_cap = jaw.capitalize()
    src = patient_dir / f"{pid}_DDC_{jaw_cap}_teeth.ply"
    if not src.is_file():
        src = patient_dir / f"{pid}_DDC_{jaw_cap}_teeth.obj"
    tgt = patient_dir / "ct_seg" / f"hi_{jaw}_teeth.stl"
    if not tgt.is_file():
        tgt = patient_dir / f"{pid}_CT_{jaw_cap}.stl"
    if not src.is_file() or not tgt.is_file():
        return None
    src_mesh = o3d.io.read_triangle_mesh(str(src))
    tgt_mesh = o3d.io.read_triangle_mesh(str(tgt))
    if src_mesh.is_empty() or tgt_mesh.is_empty():
        return None
    return src, tgt, src_mesh, tgt_mesh


def write_outputs(
    *,
    patient_dir: Path,
    pid: str,
    jaw: str,
    method: str,
    T: np.ndarray,
    src_path: Path,
    tgt_path: Path,
    label: str,
    reg_fit: float,
    reg_rmse: float,
    eval_fit: float,
    eval_rmse: float,
    status: str,
    message: str,
    save_aligned_source: bool,
) -> Path:
    import open3d as o3d

    out_dir = patient_dir / RESULT_SUBDIR[method]
    out_dir.mkdir(parents=True, exist_ok=True)
    tf_path = out_dir / f"temp_ddc_to_ct_transform_{jaw}.json"
    tf_path.write_text(json.dumps(T.tolist(), indent=2), encoding="utf-8")
    log_path = out_dir / f"registration_log_{jaw}.txt"
    log_path.write_text(
        "\n".join(
            [
                f"method: {method}",
                f"patient: {pid}",
                f"jaw: {jaw}",
                f"source: {src_path}",
                f"target: {tgt_path}",
                f"selected_init: {label}",
                f"registration_downsample_fitness: {reg_fit}",
                f"registration_downsample_rmse: {reg_rmse}",
                f"unified_delta_eval_fitness: {eval_fit}",
                f"unified_delta_eval_rmse: {eval_rmse}",
                f"status: {status}",
                f"message: {message}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if save_aligned_source:
        mesh = o3d.io.read_triangle_mesh(str(src_path))
        mesh.transform(T)
        o3d.io.write_triangle_mesh(str(out_dir / f"{pid}_{method}_aligned_{jaw}_ddc.ply"), mesh)
    return tf_path


def finite_or_inf(value: float) -> str:
    return f"{value:.17g}" if math.isfinite(value) else "inf"
