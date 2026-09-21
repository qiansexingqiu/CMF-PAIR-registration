#!/usr/bin/env python3
"""Run CMF-PAIR traditional registration methods on a patient dataset."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfpair.dataset import (
    CSV_NAME,
    delta_tag,
    finite_or_inf,
    iter_patients,
    load_pair,
    write_outputs,
)
from cmfpair.registration import METHODS, register_meshes

try:
    import tqdm as _tqdm_module

    _TQDM_AVAILABLE = True
except ImportError:
    _tqdm_module = None
    _TQDM_AVAILABLE = False


def _iter_progress(items: list[Path], *, enabled: bool, desc: str) -> Iterable[Path]:
    if enabled and _TQDM_AVAILABLE:
        return _tqdm_module.tqdm(items, desc=desc, unit="patient", dynamic_ncols=True)
    return items


def _log(message: str, *, progress: bool) -> None:
    if progress and _TQDM_AVAILABLE:
        _tqdm_module.tqdm.write(message)
    else:
        print(message, flush=True)


def parse_jaws(value: str) -> tuple[str, ...]:
    out = []
    for item in value.split(","):
        x = item.strip().lower()
        if not x:
            continue
        if x in ("upper", "u", "maxilla"):
            out.append("upper")
        elif x in ("lower", "l", "mandible"):
            out.append("lower")
        else:
            raise argparse.ArgumentTypeError(f"Invalid jaw: {item!r}")
    return tuple(out) or ("upper", "lower")


def run_method_on_jaw(
    *,
    patient_dir: Path,
    pid: str,
    jaw: str,
    method: str,
    voxel_mm: float,
    coarse_mm: float,
    fine_mm: float,
    delta: float,
    save_aligned_source: bool,
):
    pair = load_pair(patient_dir, pid, jaw)
    if pair is None:
        return None
    src_path, tgt_path, src_mesh, tgt_mesh = pair
    start = time.time()
    try:
        result = register_meshes(
            src_mesh,
            tgt_mesh,
            method,
            voxel_mm=voxel_mm,
            coarse_mm=coarse_mm,
            fine_mm=fine_mm,
            delta=delta,
        )
        T = result["transform"]
        status = "Success"
        message = result["message"]
        eval_fit = result["fitness"]
        eval_rmse = result["inlier_rmse"]
        label = result["selected_init"]
        reg_fit = result["registration_fitness_downsample"]
        reg_rmse = result["registration_rmse_downsample"]
    except Exception as exc:
        import numpy as np

        T = np.eye(4, dtype=np.float64)
        status = "Failed"
        message = str(exc)
        eval_fit = 0.0
        eval_rmse = 0.0
        label = "failed"
        reg_fit = -1.0
        reg_rmse = float("inf")

    tf_path = write_outputs(
        patient_dir=patient_dir,
        pid=pid,
        jaw=jaw,
        method=method,
        T=T,
        src_path=src_path,
        tgt_path=tgt_path,
        label=label,
        reg_fit=reg_fit,
        reg_rmse=reg_rmse,
        eval_fit=eval_fit,
        eval_rmse=eval_rmse,
        status=status,
        message=message,
        save_aligned_source=save_aligned_source,
    )
    rel_root = patient_dir.parent
    return {
        "Patient": pid,
        "jaw": jaw,
        "mode": method,
        "delta": f"{delta:.17g}",
        "fitness": f"{eval_fit:.17g}",
        "inlier_rmse": f"{eval_rmse:.17g}",
        "source": str(src_path.relative_to(rel_root)),
        "target": str(tgt_path.relative_to(rel_root)),
        "transform": str(tf_path.relative_to(rel_root)),
        "registration_fitness_downsample": f"{reg_fit:.17g}",
        "registration_rmse_downsample": finite_or_inf(reg_rmse),
        "status": status,
        "message": message,
        "elapsed_sec": f"{time.time() - start:.3f}",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run CMF-PAIR registration methods on Patient* folders.")
    ap.add_argument("-d", "--data-dir", type=Path, required=True, help="Root containing Patient* folders.")
    ap.add_argument("--method", choices=("all",) + METHODS, default="all")
    ap.add_argument("--patients", type=str, default=None, help="Comma-separated case names, e.g. Case1,Case2.")
    ap.add_argument("--jaws", type=parse_jaws, default=("upper", "lower"))
    ap.add_argument("--delta", type=float, default=1.0, help="Unified evaluation correspondence distance (mm).")
    ap.add_argument("--voxel-mm", type=float, default=0.8)
    ap.add_argument("--coarse-mm", type=float, default=5.0)
    ap.add_argument("--fine-mm", type=float, default=1.0)
    ap.add_argument("--output-dir", type=Path, default=None, help="CSV output directory; default is data-dir.")
    ap.add_argument("--append-csv", action="store_true")
    ap.add_argument("--save-aligned-source", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    root = args.data_dir.resolve()
    if not root.is_dir():
        print(f"Data directory does not exist: {root}", file=sys.stderr)
        return 2
    output_dir = (args.output_dir or root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = METHODS if args.method == "all" else (args.method,)
    only = {x.strip() for x in args.patients.split(",") if x.strip()} if args.patients else None
    patients = iter_patients(root, only)

    show_progress = not args.no_progress
    if show_progress and not _TQDM_AVAILABLE:
        print("[info] tqdm not installed; progress bar disabled. pip install tqdm")

    fields = [
        "Patient",
        "jaw",
        "mode",
        "delta",
        "fitness",
        "inlier_rmse",
        "source",
        "target",
        "transform",
        "registration_fitness_downsample",
        "registration_rmse_downsample",
        "status",
        "message",
        "elapsed_sec",
    ]

    for method in methods:
        rows = []
        desc = f"{method} delta={args.delta:g}"
        for patient_dir in _iter_progress(patients, enabled=show_progress, desc=desc):
            for jaw in args.jaws:
                row = run_method_on_jaw(
                    patient_dir=patient_dir,
                    pid=patient_dir.name,
                    jaw=jaw,
                    method=method,
                    voxel_mm=args.voxel_mm,
                    coarse_mm=args.coarse_mm,
                    fine_mm=args.fine_mm,
                    delta=args.delta,
                    save_aligned_source=args.save_aligned_source,
                )
                if row is None:
                    _log(f"[skip] {patient_dir.name} {jaw}: missing source/target", progress=show_progress)
                    continue
                rows.append(row)
                _log(
                    f"[{method}] {patient_dir.name} {jaw}: fitness={row['fitness']} "
                    f"rmse={row['inlier_rmse']} status={row['status']}",
                    progress=show_progress,
                )

        csv_path = output_dir / CSV_NAME[method].format(delta_tag=delta_tag(args.delta))
        append = bool(args.append_csv and csv_path.is_file() and csv_path.stat().st_size > 0)
        with csv_path.open("a" if append else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not append:
                w.writeheader()
            w.writerows(rows)
        print(f"{'Appended' if append else 'Wrote'} {len(rows)} rows -> {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
