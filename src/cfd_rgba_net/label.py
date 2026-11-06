"""
Label rules v1 on the voxel grid (the mesh's own axes; the up axis comes from the case geometry).

    vortex        : lambda2* < -tau_lambda2 AND Q* > 0     AND core
    shear_layer   : |omega|* > tau_omega AND NOT vortex     AND core
where core = fluid eroded by wall_erosion_H AND height above the ground >= floor_exclude_H * H.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

from .config import LabelRules
from .geometry import CaseGeometry, ImageSpec

CLASS_NAMES = ("vortex", "shear_layer")


@dataclass
class LabelResult:
    labels: np.ndarray        # (2, nx, ny, nz) uint8, CLASS_NAMES order
    core: np.ndarray          # (nx, ny, nz) bool, where labels may be set
    fluid_eroded: np.ndarray  # (nx, ny, nz) bool
    fractions: dict           # per class, over fluid voxels


def make_labels(ing: dict[str, np.ndarray], fluid: np.ndarray, spec: ImageSpec, geo: CaseGeometry,
                rules: LabelRules, n_per_H: int) -> LabelResult:
    """
    ing: resampled ingredients {Q_nd, lambda2_nd, omega_mag_nd} on spec's grid.
    fluid: voxels inside the mesh and outside the body.
    """
    erosion = rules.wall_erosion_voxels(n_per_H)
    fluid_eroded = ndi.binary_erosion(fluid, iterations=erosion) if erosion > 0 else fluid.copy()
    above_floor = geo.height_above_ground(spec) >= rules.floor_exclude_H * geo.H
    core = fluid_eroded & above_floor

    vortex = (ing["lambda2_nd"] < -rules.tau_lambda2) & (ing["Q_nd"] > 0) & core
    shear = (ing["omega_mag_nd"] > rules.tau_omega) & ~vortex & core

    labels = np.stack([vortex, shear]).astype(np.uint8)
    n_fluid = max(int(fluid.sum()), 1)
    fractions = {c: float(labels[i][fluid].sum() / n_fluid) for i, c in enumerate(CLASS_NAMES)}
    fractions["none"] = float(((labels.sum(0) == 0) & fluid).sum() / n_fluid)
    return LabelResult(labels, core, fluid_eroded, fractions)


def describe(rules: LabelRules, n_per_H: int) -> dict[str, str]:
    """The rules as text, with this configuration's numbers, for the metadata file."""
    r = rules
    core = f"fluid eroded by {r.wall_erosion_voxels(n_per_H)} voxel(s) & height above ground >= {r.floor_exclude_H} H"
    return dict(
        vortex=f"lambda2* < -{r.tau_lambda2} & Q* > 0 & core",
        shear_layer=f"|omega|* > {r.tau_omega} & ~vortex & core",
        core=core,
    )
