#!/usr/bin/env python3
"""Register a single source mesh to a target mesh and write a 4x4 transform."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfpair.registration import METHODS, register_meshes


def main() -> int:
    ap = argparse.ArgumentParser(description="Register one dental model (T1/T2) to a CT tooth mesh.")
    ap.add_argument("--source", type=Path, required=True, help="Source mesh (.ply/.obj/.stl).")
    ap.add_argument("--target", type=Path, required=True, help="Target mesh (.ply/.obj/.stl).")
    ap.add_argument("--method", choices=METHODS, default="pca_icp")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON 4x4 transform.")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--voxel-mm", type=float, default=0.8)
    ap.add_argument("--coarse-mm", type=float, default=5.0)
    ap.add_argument("--fine-mm", type=float, default=1.0)
    ap.add_argument("--save-aligned", type=Path, default=None, help="Optional aligned source mesh path.")
    args = ap.parse_args()

    import open3d as o3d

    src = o3d.io.read_triangle_mesh(str(args.source))
    tgt = o3d.io.read_triangle_mesh(str(args.target))
    if src.is_empty() or tgt.is_empty():
        print("Failed to load source or target mesh.", file=sys.stderr)
        return 2

    result = register_meshes(
        src,
        tgt,
        args.method,
        voxel_mm=args.voxel_mm,
        coarse_mm=args.coarse_mm,
        fine_mm=args.fine_mm,
        delta=args.delta,
    )
    T = result["transform"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": args.method,
        "selected_init": result["selected_init"],
        "fitness": result["fitness"],
        "inlier_rmse": result["inlier_rmse"],
        "transform": T.tolist(),
        "source": str(args.source),
        "target": str(args.target),
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.save_aligned is not None:
        aligned = o3d.io.read_triangle_mesh(str(args.source))
        aligned.transform(T)
        o3d.io.write_triangle_mesh(str(args.save_aligned), aligned)
    print(
        f"{args.method}: fitness={result['fitness']:.4f} rmse={result['inlier_rmse']:.4f} "
        f"init={result['selected_init']} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
