#!/usr/bin/env python3
"""Path-resolution tests for the CMF-PAIR Zenodo / paper folder layout."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmfpair.dataset import (
    find_source_mesh,
    find_target_mesh,
    iter_patients,
    resolve_pair_paths,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"solid dummy\nendsolid dummy\n")


def test_zenodo_case_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = root / "Case1"
        _touch(case / "T1.stl")
        _touch(case / "T2.stl")
        _touch(case / "Segmentation_Upper Teeth.stl")
        _touch(case / "Segmentation_Lower Teeth.stl")
        _touch(case / "ct.nii.gz")

        patients = iter_patients(root)
        assert [p.name for p in patients] == ["Case1"], patients
        assert iter_patients(root, {"1"})[0].name == "Case1"
        assert iter_patients(root, {"Case1"})[0].name == "Case1"

        upper = resolve_pair_paths(case, "upper")
        lower = resolve_pair_paths(case, "lower")
        assert upper is not None and upper[0].name == "T1.stl"
        assert upper[1].name == "Segmentation_Upper Teeth.stl"
        assert lower is not None and lower[0].name == "T2.stl"
        assert lower[1].name == "Segmentation_Lower Teeth.stl"


def test_segmentations_subfolder_and_underscore_names() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp) / "Case12"
        _touch(case / "T1.stl")
        _touch(case / "T2.stl")
        _touch(case / "Segmentations" / "Segmentation_Upper_Teeth.stl")
        _touch(case / "Segmentations" / "Segmentation_Lower_Teeth.stl")
        assert find_source_mesh(case, "upper").name == "T1.stl"
        assert find_target_mesh(case, "upper").name == "Segmentation_Upper_Teeth.stl"
        assert find_target_mesh(case, "lower").name == "Segmentation_Lower_Teeth.stl"


def test_nested_dataset_root_and_single_case_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "Dataset"
        _touch(root / "Case2" / "T1.stl")
        _touch(root / "Case2" / "T2.stl")
        _touch(root / "Case2" / "Segmentation_Upper Teeth.stl")
        _touch(root / "Case2" / "Segmentation_Lower Teeth.stl")
        _touch(root / "Case10" / "T1.stl")
        _touch(root / "Case10" / "T2.stl")
        _touch(root / "Case10" / "Segmentation_Upper Teeth.stl")
        _touch(root / "Case10" / "Segmentation_Lower Teeth.stl")
        names = [p.name for p in iter_patients(root)]
        assert names == ["Case2", "Case10"], names
        assert iter_patients(root / "Case2")[0].name == "Case2"


def test_perturb_dir_is_not_used_as_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp) / "Case3"
        _touch(case / "T1.stl")
        _touch(case / "Segmentation_Upper Teeth.stl")
        _touch(case / "perturb_heavy" / "T1.stl")
        src = find_source_mesh(case, "upper")
        assert src is not None
        assert src.parent.name == "Case3"


def test_legacy_patientddc_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp) / "PatientDDC_case8"
        _touch(case / "PatientDDC_case8_DDC_Upper_teeth.ply")
        _touch(case / "PatientDDC_case8_DDC_Lower_teeth.ply")
        _touch(case / "ct_seg" / "hi_upper_teeth.stl")
        _touch(case / "ct_seg" / "hi_lower_teeth.stl")
        patients = iter_patients(Path(tmp))
        assert [p.name for p in patients] == ["PatientDDC_case8"]
        upper = resolve_pair_paths(case, "upper")
        assert upper is not None
        assert upper[0].name.endswith("_DDC_Upper_teeth.ply")
        assert upper[1].name == "hi_upper_teeth.stl"


if __name__ == "__main__":
    tests = [
        test_zenodo_case_layout,
        test_segmentations_subfolder_and_underscore_names,
        test_nested_dataset_root_and_single_case_dir,
        test_perturb_dir_is_not_used_as_source,
        test_legacy_patientddc_layout,
    ]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print("all passed")
