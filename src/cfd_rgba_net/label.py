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
    """The labels of one case, with the mask they were restricted to.

    Attributes
    ----------
    labels : numpy.ndarray
        Shape ``(2, nx, ny, nz)``, dtype uint8, 1 where a class applies. Channels follow ``CLASS_NAMES``.
    core : numpy.ndarray
        Boolean ``(nx, ny, nz)``: where labels may be set.
    fluid_eroded : numpy.ndarray
        Boolean ``(nx, ny, nz)``: the fluid mask after the wall erosion.
    fractions : dict[str, float]
        Fraction of the fluid voxels in each class, and under ``"none"`` those with no label.
    """
    labels: np.ndarray
    core: np.ndarray
    fluid_eroded: np.ndarray
    fractions: dict


def make_labels(ing: dict[str, np.ndarray], fluid: np.ndarray, spec: ImageSpec, geo: CaseGeometry,
                rules: LabelRules, n_per_H: int) -> LabelResult:
    """Apply the label rules to the ingredients on the voxel grid.

    The rules are in the module docstring. Their text in ``describe`` repeats them: keep the two in step,
    and bump ``LABEL_RULES_VERSION`` when a rule changes.

    The wall erosion removes the fluid voxels within ``rules.wall_erosion_voxels(n_per_H)`` voxels of a
    non-fluid one, with face-connected (6-neighbour) steps, so diagonal neighbours do not erode. It also
    treats everything outside the grid as non-fluid: no label is ever set on the grid's outermost layers.
    That is intended where the grid fills the mesh's domain, whose faces are mesh boundaries.

    Parameters
    ----------
    ing : dict[str, numpy.ndarray]
        The resampled ingredients ``Q_nd``, ``lambda2_nd`` and ``omega_mag_nd``, each of shape
        ``spec.dims``. Non-finite values never satisfy a rule.
    fluid : numpy.ndarray
        Boolean array of shape ``spec.dims``: voxels inside the mesh and outside the body.
    spec : ImageSpec
        The voxel grid.
    geo : CaseGeometry
        Body and ground geometry. Supplies H and the height above the ground.
    rules : LabelRules
        The label thresholds.
    n_per_H : int
        Voxels per reference length H, from the contract.

    Returns
    -------
    LabelResult
        The labels, the mask they are restricted to, and the class fractions. The fractions count over
        all fluid voxels, not only the core: fluid voxels removed by the erosion or the floor rule are
        in ``"none"``.
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
    """Write the rules as text, with this configuration's numbers, for the metadata file.

    The text repeats the rules in ``make_labels``: keep the two in step.

    Parameters
    ----------
    rules : LabelRules
        The label thresholds.
    n_per_H : int
        Voxels per reference length H, from the contract. Converts the wall erosion to voxels.

    Returns
    -------
    dict[str, str]
        The rule of each class under its name in ``CLASS_NAMES``, and the definition of ``core``.
    """
    r = rules
    core = f"fluid eroded by {r.wall_erosion_voxels(n_per_H)} voxel(s) & height above ground >= {r.floor_exclude_H} H"
    return dict(
        vortex=f"lambda2* < -{r.tau_lambda2} & Q* > 0 & core",
        shear_layer=f"|omega|* > {r.tau_omega} & ~vortex & core",
        core=core,
    )
