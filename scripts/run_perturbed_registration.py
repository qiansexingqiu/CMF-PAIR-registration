#!/usr/bin/env python3
"""Perturbed-source registration benchmark for CMF-PAIR.

T1/T2 are ProPlan-exported dental models already in CT space. A known rigid
perturbation is applied, then methods register the moved source back to the
unchanged CT dentition. Ground truth is T_gt = inv(T_pert). There is no
unperturbed ("none") level.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfpair.dataset import (
    RESULT_SUBDIR,
    delta_tag,
    find_source_mesh,
    find_target_mesh,
    finite_or_inf,
    iter_patients,
    load_pair,
    patient_sort_key,
)
from cmfpair.registration import METHODS, register_meshes

PERTURB_LEVELS = {
    "light": {"trans_mm": (10.0, 30.0), "rot_deg": (5.0, 15.0)},
    "medium": {"trans_mm": (50.0, 100.0), "rot_deg": (30.0, 60.0)},
    "heavy": {"trans_mm": (200.0, 400.0), "rot_deg": (90.0, 180.0)},
}


def perturb_dir(patient_dir: Path, level: str) -> Path:
    return patient_dir / f"perturb_{level}"


def _rng_for(patient_id: str, jaw: str, level: str, base_seed: int) -> np.random.Generator:
    from pathlib import Path

    jaw_code = 1 if jaw == "upper" else 2
    level_code = {"light": 11, "medium": 22, "heavy": 33}[level]
    case_num = patient_sort_key(Path(patient_id))
    return np.random.default_rng(base_seed + case_num * 1000 + jaw_code + level_code)


def _rotation_matrix_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ],
        dtype=np.float64,
    )


def sample_perturbation(rng: np.random.Generator, level: str) -> tuple[np.ndarray, dict[str, Any]]:
    spec = PERTURB_LEVELS[level]
    t_lo, t_hi = spec["trans_mm"]
    r_lo, r_hi = spec["rot_deg"]
    trans_norm = float(rng.uniform(t_lo, t_hi))
    rot_deg = float(rng.uniform(r_lo, r_hi))
    direction = rng.normal(size=3)
    direction /= max(np.linalg.norm(direction), 1e-12)
    translation = direction * trans_norm
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-12)
    R = _rotation_matrix_axis_angle(axis, math.radians(rot_deg))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = translation
    meta = {
        "level": level,
        "translation_mm": trans_norm,
        "rotation_deg": rot_deg,
        "translation_vec": translation.tolist(),
        "rotation_axis": axis.tolist(),
    }
    return T, meta


def pose_errors(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    T_err = T_est @ np.linalg.inv(T_gt)
    R = T_err[:3, :3]
    trace = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    rot_deg = math.degrees(math.acos(trace))
    trans_mm = float(np.linalg.norm(T_err[:3, 3]))
    return rot_deg, trans_mm


def _perturbed_source_path(pert_dir: Path, original_src: Path) -> Path:
    """Keep the Zenodo filename (T1.stl / T2.stl) inside perturb_{level}/."""
    return pert_dir / original_src.name


def _write_perturbed_mesh(src_mesh, dest: Path) -> None:
    import open3d as o3d

    dest.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if dest.suffix.lower() in {".ply", ".obj"}:
        kwargs["write_vertex_colors"] = True
    o3d.io.write_triangle_mesh(str(dest), src_mesh, **kwargs)


def parse_jaws(value: str) -> tuple[str, ...]:
    out = []
    for item in value.split(","):
        x = item.strip().lower()
        if x in ("upper", "u", "maxilla"):
            out.append("upper")
        elif x in ("lower", "l", "mandible"):
            out.append("lower")
        elif x:
            raise argparse.ArgumentTypeError(f"Invalid jaw: {item!r}")
    return tuple(out) or ("upper", "lower")


def parse_levels(value: str) -> tuple[str, ...]:
    out = []
    for item in value.split(","):
        x = item.strip().lower()
        if not x:
            continue
        if x not in PERTURB_LEVELS:
            raise argparse.ArgumentTypeError(f"Unknown level {item!r}")
        out.append(x)
    return tuple(out) or tuple(PERTURB_LEVELS.keys())


def parse_methods(value: str) -> tuple[str, ...]:
    out = []
    for item in value.split(","):
        x = item.strip()
        if not x:
            continue
        if x not in METHODS:
            raise argparse.ArgumentTypeError(f"Unknown method {item!r}")
        out.append(x)
    return tuple(out) or METHODS


def cmd_prepare(args) -> int:
    root = args.data_dir.resolve()
    only = {x.strip() for x in args.patients.split(",") if x.strip()} if args.patients else None
    patients = iter_patients(root, only)

    for level in args.levels:
        print(f"\n{'=' * 60}\n[prepare] level={level}\n{'=' * 60}", flush=True)
        for patient_dir in patients:
            pid = patient_dir.name
            pert = perturb_dir(patient_dir, level)
            pert.mkdir(parents=True, exist_ok=True)
            manifest: dict[str, Any] = {
                "patient": pid,
                "level": level,
                "seed_base": args.seed,
                "perturb_spec": PERTURB_LEVELS[level],
                "jaws": {},
            }
            for jaw in args.jaws:
                pair = load_pair(patient_dir, pid, jaw)
                if pair is None:
                    print(f"  [skip] {pid} {jaw}: missing original source/target", flush=True)
                    continue
                orig_src_path, tgt_path, src_mesh, tgt_mesh = pair
                rng = _rng_for(pid, jaw, level, args.seed)
                T_pert, pmeta = sample_perturbation(rng, level)
                T_gt = np.linalg.inv(T_pert)
                src_pert = copy.deepcopy(src_mesh)
                src_pert.transform(T_pert)
                out_src = _perturbed_source_path(pert, orig_src_path)
                _write_perturbed_mesh(src_pert, out_src)
                tf_path = pert / f"perturb_transform_{jaw}.json"
                tf_path.write_text(
                    json.dumps(
                        {
                            "patient": pid,
                            "jaw": jaw,
                            "T_pert": T_pert.tolist(),
                            "T_gt": T_gt.tolist(),
                            "ground_truth": "ProPlan-exported T1/T2 pose (identity in CT space before perturbation)",
                            **pmeta,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                c0 = float(np.linalg.norm(np.asarray(src_mesh.vertices).mean(0) - np.asarray(tgt_mesh.vertices).mean(0)))
                c1 = float(np.linalg.norm(np.asarray(src_pert.vertices).mean(0) - np.asarray(tgt_mesh.vertices).mean(0)))
                manifest["jaws"][jaw] = {
                    "original_source": str(orig_src_path.relative_to(root)),
                    "perturbed_source": str(out_src.relative_to(root)),
                    "target": str(tgt_path.relative_to(root)),
                    "center_dist_before_mm": c0,
                    "center_dist_after_mm": c1,
                    **pmeta,
                }
                print(
                    f"  {pid} {jaw}: trans={pmeta['translation_mm']:.1f}mm rot={pmeta['rotation_deg']:.1f}deg "
                    f"| center {c0:.1f}->{c1:.1f}mm",
                    flush=True,
                )
            if manifest["jaws"]:
                (pert / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


def _load_perturbed_pair(patient_dir: Path, pid: str, jaw: str, level: str):
    import open3d as o3d

    pert = perturb_dir(patient_dir, level)
    src = find_source_mesh(pert, jaw, pid)
    src_mesh = None
    if src is not None:
        mesh = o3d.io.read_triangle_mesh(str(src))
        if not mesh.is_empty():
            src_mesh = mesh
    if src is None or src_mesh is None:
        return None

    tgt = find_target_mesh(patient_dir, jaw, pid)
    tf_path = pert / f"perturb_transform_{jaw}.json"
    if tgt is None or not tf_path.is_file():
        return None
    tgt_mesh = o3d.io.read_triangle_mesh(str(tgt))
    if tgt_mesh.is_empty():
        return None
    meta = json.loads(tf_path.read_text(encoding="utf-8"))
    T_gt = np.array(meta["T_gt"], dtype=np.float64)
    return src, tgt, src_mesh, tgt_mesh, T_gt, meta


def cmd_register(args) -> int:
    root = args.data_dir.resolve()
    output_dir = (args.output_dir or root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in args.patients.split(",") if x.strip()} if args.patients else None
    patients = iter_patients(root, only)
    fields = [
        "Patient",
        "jaw",
        "perturb_level",
        "mode",
        "delta",
        "fitness",
        "inlier_rmse",
        "gt_rotation_err_deg",
        "gt_translation_err_mm",
        "perturb_translation_mm",
        "perturb_rotation_deg",
        "perturbed_source",
        "target",
        "transform",
        "registration_fitness_downsample",
        "registration_rmse_downsample",
        "selected_init",
        "status",
        "message",
        "elapsed_sec",
    ]
    rows_by_level: dict[str, list[dict]] = {lvl: [] for lvl in args.levels}

    for level in args.levels:
        print(f"\n{'=' * 60}\n[register] level={level} delta={args.delta}\n{'=' * 60}", flush=True)
        for patient_dir in patients:
            pid = patient_dir.name
            if not perturb_dir(patient_dir, level).is_dir():
                print(f"  [skip] {pid}: missing perturb_{level}/", flush=True)
                continue
            for jaw in args.jaws:
                loaded = _load_perturbed_pair(patient_dir, pid, jaw, level)
                if loaded is None:
                    print(f"  [skip] {pid} {jaw}: missing perturbed source/target/transform", flush=True)
                    continue
                src_path, tgt_path, src_mesh, tgt_mesh, T_gt, pmeta = loaded
                for method in args.methods:
                    method_dir = perturb_dir(patient_dir, level) / RESULT_SUBDIR[method]
                    method_dir.mkdir(parents=True, exist_ok=True)
                    start = time.time()
                    try:
                        result = register_meshes(
                            src_mesh,
                            tgt_mesh,
                            method,
                            voxel_mm=args.voxel_mm,
                            coarse_mm=args.coarse_mm,
                            fine_mm=args.fine_mm,
                            delta=args.delta,
                        )
                        T_est = result["transform"]
                        eval_fit = result["fitness"]
                        eval_rmse = result["inlier_rmse"]
                        rot_err, trans_err = pose_errors(T_est, T_gt)
                        status = "Success"
                        message = result["message"]
                        label = result["selected_init"]
                        reg_fit = result["registration_fitness_downsample"]
                        reg_rmse = result["registration_rmse_downsample"]
                    except Exception as exc:
                        T_est = np.eye(4)
                        eval_fit = eval_rmse = 0.0
                        rot_err = trans_err = float("nan")
                        status = "Failed"
                        message = str(exc)
                        label = "failed"
                        reg_fit = -1.0
                        reg_rmse = float("inf")
                    tf_path = method_dir / f"temp_ddc_to_ct_transform_{jaw}.json"
                    tf_path.write_text(json.dumps(T_est.tolist(), indent=2), encoding="utf-8")
                    row = {
                        "Patient": pid,
                        "jaw": jaw,
                        "perturb_level": level,
                        "mode": method,
                        "delta": f"{args.delta:.17g}",
                        "fitness": f"{eval_fit:.17g}",
                        "inlier_rmse": f"{eval_rmse:.17g}",
                        "gt_rotation_err_deg": f"{rot_err:.17g}" if math.isfinite(rot_err) else "nan",
                        "gt_translation_err_mm": f"{trans_err:.17g}" if math.isfinite(trans_err) else "nan",
                        "perturb_translation_mm": f"{pmeta['translation_mm']:.17g}",
                        "perturb_rotation_deg": f"{pmeta['rotation_deg']:.17g}",
                        "perturbed_source": str(src_path.relative_to(root)),
                        "target": str(tgt_path.relative_to(root)),
                        "transform": str(tf_path.relative_to(root)),
                        "registration_fitness_downsample": f"{reg_fit:.17g}",
                        "registration_rmse_downsample": finite_or_inf(reg_rmse),
                        "selected_init": label,
                        "status": status,
                        "message": message,
                        "elapsed_sec": f"{time.time() - start:.3f}",
                    }
                    rows_by_level[level].append(row)
                    print(
                        f"    [{method}] {pid} {jaw}: fitness={eval_fit:.4f} rmse={eval_rmse:.4f} "
                        f"| pose rot={rot_err:.2f}deg trans={trans_err:.2f}mm",
                        flush=True,
                    )

    tag = delta_tag(args.delta)
    for level, rows in rows_by_level.items():
        csv_path = output_dir / f"unified_fitness_perturb_{level}_delta{tag}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {csv_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Perturbed T1/T2 registration benchmark (ProPlan-exported pose as GT)."
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-d", "--data-dir", type=Path, required=True)
        p.add_argument("--patients", type=str, default=None)
        p.add_argument("--jaws", type=parse_jaws, default=("upper", "lower"))
        p.add_argument("--levels", type=parse_levels, default=tuple(PERTURB_LEVELS.keys()))
        p.add_argument("--seed", type=int, default=42)

    def add_register(p: argparse.ArgumentParser) -> None:
        p.add_argument("--methods", type=parse_methods, default=METHODS)
        p.add_argument("--delta", type=float, default=1.0)
        p.add_argument("--voxel-mm", type=float, default=0.8)
        p.add_argument("--coarse-mm", type=float, default=5.0)
        p.add_argument("--fine-mm", type=float, default=1.0)
        p.add_argument("--output-dir", type=Path, default=None)

    p_prep = sub.add_parser("prepare", help="Write perturbed meshes into perturb_{level}/")
    add_common(p_prep)
    p_reg = sub.add_parser("register", help="Register meshes in perturb_{level}/")
    add_common(p_reg)
    add_register(p_reg)
    p_all = sub.add_parser("all", help="prepare then register")
    add_common(p_all)
    add_register(p_all)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "register":
        return cmd_register(args)
    if args.command == "all":
        cmd_prepare(args)
        return cmd_register(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
