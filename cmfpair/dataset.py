"""Dataset folder conventions used by CMF-PAIR registration scripts.

Primary layout matches the CMF-PAIR paper / Zenodo release:

    Case1/
      T1.stl                          # maxillary dental model (ProPlan-exported, CT space)
      T2.stl                          # mandibular dental model
      Segmentation_Upper Teeth.stl    # CT maxillary dentition
      Segmentation_Lower Teeth.stl    # CT mandibular dentition

T1/T2 are the clinically registered dental models exported from ProPlan CMF.
They already sit in the CT coordinate system and are the pose ground truth
for the perturbation benchmark (light / medium / heavy only).

Legacy ``PatientDDC_*`` folders with ``*_DDC_*_teeth`` and ``ct_seg/hi_*_teeth.stl``
are still accepted as a fallback.
"""
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

_CASE_DIR_RE = re.compile(r"^case\d+$", re.I)
_PATIENT_DIR_RE = re.compile(r"^patient", re.I)
_SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "weights",
    *RESULT_SUBDIR.values(),
    "traditional_single_pca_icp_result",
    "traditional_pca_icp_result",
    "traditional_fpfh_ransac_icp_result",
}


def _norm_key(name: str) -> str:
    text = Path(name).name.lower().replace("&", "and")
    text = re.sub(r"[_\-\s]+", " ", text).strip()
    return text


def _skip_dir_name(name: str) -> bool:
    lowered = name.lower()
    if lowered.startswith("perturb_"):
        return True
    return name in _SKIP_DIR_NAMES or lowered in {x.lower() for x in _SKIP_DIR_NAMES}


def source_mesh_names(pid: str, jaw: str) -> list[str]:
    jaw_cap = jaw.capitalize()
    if jaw == "upper":
        names = ["T1.stl", "T1.ply", "T1.obj"]
    else:
        names = ["T2.stl", "T2.ply", "T2.obj"]
    names += [
        f"{pid}_DDC_{jaw_cap}_teeth.ply",
        f"{pid}_DDC_{jaw_cap}_teeth.obj",
        f"{pid}_DDC_{jaw_cap}_teeth.stl",
    ]
    return names


def target_mesh_names(pid: str, jaw: str) -> list[str]:
    jaw_cap = jaw.capitalize()
    if jaw == "upper":
        names = [
            "Segmentation_Upper Teeth.stl",
            "Segmentation_Upper_Teeth.stl",
        ]
    else:
        names = [
            "Segmentation_Lower Teeth.stl",
            "Segmentation_Lower_Teeth.stl",
        ]
    names += [
        f"ct_seg/hi_{jaw}_teeth.stl",
        f"hi_{jaw}_teeth.stl",
        f"{pid}_CT_{jaw_cap}.stl",
    ]
    return names


def _iter_case_files(case_dir: Path):
    for path in case_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(case_dir).parts
        if any(_skip_dir_name(part) for part in rel_parts[:-1]):
            continue
        yield path


def find_mesh(case_dir: Path, names: list[str]) -> Path | None:
    """Locate a mesh by paper filename first, then by normalized basename."""
    for name in names:
        path = case_dir / name
        if path.is_file():
            return path

    order = {_norm_key(Path(name).name): i for i, name in enumerate(names)}
    best: Path | None = None
    best_rank = (10**9, 10**9)
    for path in _iter_case_files(case_dir):
        key = _norm_key(path.name)
        if key not in order:
            continue
        rank = (order[key], len(path.relative_to(case_dir).parts))
        if rank < best_rank:
            best = path
            best_rank = rank
    return best


def find_source_mesh(case_dir: Path, jaw: str, pid: str | None = None) -> Path | None:
    return find_mesh(case_dir, source_mesh_names(pid or case_dir.name, jaw))


def find_target_mesh(case_dir: Path, jaw: str, pid: str | None = None) -> Path | None:
    return find_mesh(case_dir, target_mesh_names(pid or case_dir.name, jaw))


def resolve_pair_paths(
    patient_dir: Path, jaw: str, pid: str | None = None
) -> tuple[Path, Path] | None:
    pid = pid or patient_dir.name
    src = find_source_mesh(patient_dir, jaw, pid)
    tgt = find_target_mesh(patient_dir, jaw, pid)
    if src is None or tgt is None:
        return None
    return src, tgt


def is_case_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    if _CASE_DIR_RE.match(name):
        return True
    return bool(_PATIENT_DIR_RE.match(name) and re.search(r"\d+", name))


def patient_sort_key(path: Path) -> int:
    name = path.name
    match = re.search(r"(?i)case(\d+)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else 999999


def _matches_only(path: Path, only: set[str]) -> bool:
    lowered = {item.lower() for item in only}
    if path.name in only or path.name.lower() in lowered:
        return True
    return str(patient_sort_key(path)) in only


def iter_patients(root: Path, only: set[str] | None = None) -> list[Path]:
    """Yield Case1..Case100 (paper/Zenodo) or legacy Patient* folders."""
    root = Path(root)
    if not root.is_dir():
        return []

    candidates = [p for p in root.iterdir() if is_case_dir(p)]
    if not candidates:
        nested: list[Path] = []
        for child in root.iterdir():
            if child.is_dir():
                nested.extend(p for p in child.iterdir() if is_case_dir(p))
        candidates = nested
    if not candidates and is_case_dir(root):
        candidates = [root]

    patients = sorted(candidates, key=patient_sort_key)
    if only:
        patients = [p for p in patients if _matches_only(p, only)]
    return patients


def delta_tag(delta: float) -> str:
    if abs(delta - round(delta)) < 1e-9:
        return str(int(round(delta)))
    return str(delta).replace(".", "p")


def load_pair(patient_dir: Path, pid: str, jaw: str):
    import open3d as o3d

    resolved = resolve_pair_paths(patient_dir, jaw, pid)
    if resolved is None:
        return None
    src, tgt = resolved
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
        suffix = src_path.suffix.lower() if src_path.suffix else ".ply"
        o3d.io.write_triangle_mesh(
            str(out_dir / f"{pid}_{method}_aligned_{jaw}{suffix}"), mesh
        )
    return tf_path


def finite_or_inf(value: float) -> str:
    return f"{value:.17g}" if math.isfinite(value) else "inf"
