# CMF-PAIR Registration

Official registration code of **CMF-PAIR: A Patient-Level Multimodal Clinical Dataset for Craniofacial Registration, Reconstruction and Treatment Planning**.

This repository releases the three geometry-based jaw-level methods used in the paper:

| Paper name | CLI flag | Initialization |
|---|---|---|
| Single PCA-ICP | `single_pca_icp` | One PCA frame (`pca-000`) + coarse-to-fine ICP |
| Multi-hypothesis PCA-ICP | `pca_icp` | Four proper PCA sign hypotheses + ICP, pick best |
| FPFH RANSAC ICP | `fpfh_ransac_icp` | FPFH + RANSAC global registration + ICP |

All methods estimate a rigid transform **T** from the IOS / DDC tooth mesh to the CT tooth mesh (`T @ source ≈ target`). Reported fitness / inlier RMSE use Open3D `evaluate_registration` on **full-resolution vertices** at correspondence distance **δ = 1.0 mm**.

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

These weights are only needed if you start from raw IOS meshes and CT volumes. If `*_DDC_*_teeth` and `ct_seg/hi_*_teeth.stl` already exist, skip this section and go straight to registration.

---

## Data layout

Each patient folder should look like:

```text
<data_root>/
  PatientDDC_case001/
    PatientDDC_case001_DDC_Upper_teeth.ply   # or .obj
    PatientDDC_case001_DDC_Lower_teeth.ply
    ct_seg/
      hi_upper_teeth.stl
      hi_lower_teeth.stl
```

Fallback CT names `Patient*_CT_Upper.stl` / `Patient*_CT_Lower.stl` are also accepted.

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
  --patients PatientDDC_case001,PatientDDC_case002 \
  --jaws upper,lower \
  --delta 1.0
```

Single mesh pair:

```bash
python scripts/register_pair.py \
  --source /path/to/ddc_upper_teeth.ply \
  --target /path/to/hi_upper_teeth.stl \
  --method pca_icp \
  --out transform.json \
  --save-aligned aligned_source.ply
```

Default protocol: voxel **0.8 mm**, coarse ICP **5.0 mm**, fine ICP **1.0 mm**, evaluation δ **1.0 mm**.

Outputs per patient / method:

```text
Patient*/traditional_pca_icp_result/temp_ddc_to_ct_transform_{upper|lower}.json
<data_root>/unified_fitness_{method}_delta1.csv
```

CSV `fitness` / `inlier_rmse` are the unified full-vertex metrics. Downsampled ICP fitness is stored only for debugging.

---

## Perturbation benchmark

Source meshes are first moved by a known rigid perturbation, then registered back to the unchanged CT target. Ground-truth recovery is `T_gt = inv(T_pert)`. Seed is **42**.

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

src = o3d.io.read_triangle_mesh("ddc_upper_teeth.ply")
tgt = o3d.io.read_triangle_mesh("hi_upper_teeth.stl")
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
