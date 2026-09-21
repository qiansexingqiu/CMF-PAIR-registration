# CMF-PAIR Registration

Official registration code of **CMF-PAIR: A Patient-Level Multimodal Clinical Dataset for Craniofacial Registration, Reconstruction and Treatment Planning**.

This repository releases the three geometry-based jaw-level methods used in the paper:

| Paper name | CLI flag | Initialization |
|---|---|---|
| Single PCA-ICP | `single_pca_icp` | One PCA frame (`pca-000`) + coarse-to-fine ICP |
| Multi-hypothesis PCA-ICP | `pca_icp` | Four proper PCA sign hypotheses + ICP, pick best |
| FPFH RANSAC ICP | `fpfh_ransac_icp` | FPFH + RANSAC global registration + ICP |

All methods estimate a rigid transform **T** from the optical dental model to the CT dentition (`T @ source ≈ target`). Reported fitness / inlier RMSE use Open3D `evaluate_registration` on **full-resolution vertices** at correspondence distance **δ = 1.0 mm**.

Segmentation networks are **not** required to run these scripts if tooth meshes are already available. Pretrained segmentation weights are hosted separately (see below).

---

## Installation

```bash
git clone https://github.com/qiansexingqiu/CMF-PAIR-registration.git
cd CMF-PAIR-registration
conda create -n cmfpair python=3.10 -y
conda activate cmfpair
pip install -r requirements.txt
```

Open3D is required. A GPU is not required for the three registration methods.

---

## Segmentation weights

Weights are **not** stored in this repository. Download them from Google Drive:

1. **IOS / DDC tooth segmentation weights**  
   Download: https://drive.google.com/drive/folders/17zoALd8rb6BQfa8w8-orsrVzSIM1Xkhq?usp=drive_link  
   Place under: `weights/ios_tooth_seg/`

2. **CT craniofacial segmentation weights** (DentalSegmentator / nnU-Net, labels: maxilla & upper skull, mandible, upper teeth, lower teeth, mandibular canal)  
   Download: https://drive.google.com/drive/folders/1xJ2Upchg1ed8sA6MJvQiaaL485xIaaMu?usp=sharing  
   Place under: `weights/ct_craniofacial_seg/`

These weights are only needed if you start from raw optical scans and CT volumes. If the released Zenodo meshes are already present (`T1.stl` / `T2.stl` and `Segmentation_* Teeth.stl`), skip this section and go straight to registration.

---

## Data layout

Scripts read the **CMF-PAIR Zenodo / paper folder layout** (Fig. 5 in the manuscript):

```text
<data_root>/
  Case1/
    T1.stl                          # maxillary dental model
    T2.stl                          # mandibular dental model
    Segmentation_Upper Teeth.stl    # CT maxillary dentition
    Segmentation_Lower Teeth.stl    # CT mandibular dentition
    ct.nii.gz                       # not used by these scripts
  Case2/
    ...
  Case100/
    ...
```

`T1.stl` / `T2.stl` are the dental models **exported after ProPlan CMF registration**, already in the CT coordinate system. That pose is the ground truth for the perturbation benchmark.

Files may sit in the case root or in a subfolder such as `Segmentations/`. Names with spaces or underscores are both accepted (`Segmentation_Upper Teeth.stl` / `Segmentation_Upper_Teeth.stl`).

`--patients Case1,Case2` or `--patients 1,2` selects cases. If `-d` points at a single `Case*` folder, that case is used.

Legacy `PatientDDC_*` folders with `*_DDC_*_teeth.ply` and `ct_seg/hi_*_teeth.stl` still work.

---

## Run registration

All three methods:

```bash
python scripts/run_registration.py \
  -d /path/to/data_root \
  --method all \
  --delta 1.0
```

One method, selected patients:

```bash
python scripts/run_registration.py \
  -d /path/to/data_root \
  --method pca_icp \
  --patients Case1,Case2 \
  --jaws upper,lower \
  --delta 1.0
```

Single mesh pair:

```bash
python scripts/register_pair.py \
  --source /path/to/T1.stl \
  --target "/path/to/Segmentation_Upper Teeth.stl" \
  --method pca_icp \
  --out transform.json \
  --save-aligned aligned_source.ply
```

Default protocol: voxel **0.8 mm**, coarse ICP **5.0 mm**, fine ICP **1.0 mm**, evaluation δ **1.0 mm**.

Outputs per patient / method:

```text
Case*/traditional_pca_icp_result/temp_ddc_to_ct_transform_{upper|lower}.json
<data_root>/unified_fitness_{method}_delta1.csv
```

CSV `fitness` / `inlier_rmse` are the unified full-vertex metrics. Downsampled ICP fitness is stored only for debugging.

---

## Perturbation benchmark

`T1.stl` / `T2.stl` are first displaced by a known rigid perturbation, then registered back to the unchanged CT dentition (`Segmentation_* Teeth.stl`). Because those dental models were exported from ProPlan already in CT space, ground-truth recovery is `T_gt = inv(T_pert)`.

| Level | Translation | Rotation |
|---|---|---|
| light | 10–30 mm | 5–15° |
| medium | 50–100 mm | 30–60° |
| heavy | 200–400 mm | 90–180° |

```bash
python scripts/run_perturbed_registration.py prepare \
  -d /path/to/data_root --levels light,medium,heavy

python scripts/run_perturbed_registration.py register \
  -d /path/to/data_root \
  --levels light,medium,heavy \
  --methods single_pca_icp,pca_icp,fpfh_ransac_icp \
  --delta 1.0
```

Or both steps: `python scripts/run_perturbed_registration.py all -d /path/to/data_root`.

---

## Python API

```python
import open3d as o3d
from cmfpair import register_meshes

src = o3d.io.read_triangle_mesh("T1.stl")
tgt = o3d.io.read_triangle_mesh("Segmentation_Upper Teeth.stl")
out = register_meshes(src, tgt, method="pca_icp")
print(out["fitness"], out["inlier_rmse"])
T = out["transform"]  # 4x4 numpy array, source -> target
```

---

## Method notes

- **Single PCA-ICP** uses only the first proper PCA sign. It is the weaker PCA baseline.
- **Multi-hypothesis PCA-ICP** enumerates four determinant-+1 axis flips (`pca-000`, `pca-011`, `pca-101`, `pca-110`) and keeps the candidate with highest fitness / lowest RMSE. This is the most stable of the three methods under large pose perturbation.
- **FPFH RANSAC ICP** does not use PCA. RANSAC provides the global initial pose; ICP is point-to-point.
- Fitness can remain moderate on a 180° arch-symmetric flip. For the perturbation benchmark, also check `gt_rotation_err_deg`.

---

## License

MIT. See [LICENSE](LICENSE).
