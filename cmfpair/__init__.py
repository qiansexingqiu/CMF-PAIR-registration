"""Official registration methods of CMF-PAIR.

Methods
-------
- ``single_pca_icp``: single-hypothesis PCA initialization + ICP
- ``pca_icp``: multi-hypothesis PCA initialization + ICP
- ``fpfh_ransac_icp``: FPFH + RANSAC global registration + ICP
"""

from .registration import (
    METHODS,
    register_meshes,
    run_fpfh_ransac_icp,
    run_pca_icp,
    run_single_pca_icp,
)

__all__ = [
    "METHODS",
    "register_meshes",
    "run_fpfh_ransac_icp",
    "run_pca_icp",
    "run_single_pca_icp",
]
__version__ = "1.0.0"
