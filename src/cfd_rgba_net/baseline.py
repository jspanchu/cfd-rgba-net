"""
Voxel-grid baseline: label rules v1 applied to finite-difference gradients of the grid's own u, v, w.

It needs no training. The net must beat it (the baseline gate in the roadmap).
"""

from __future__ import annotations

import numpy as np

from .config import Contract, LabelRules
from .ingredients import lambda2, q_and_vorticity


def taps(a: np.ndarray, axis: int):
    """Give access to shifted copies of an array along one axis.

    Parameters
    ----------
    a : numpy.ndarray
        The array. It is zero-padded by 2 voxels on both sides of ``axis``.
    axis : int
        The axis to shift along.

    Returns
    -------
    callable
        ``at(k)`` returns the view of ``a`` shifted by ``k`` voxels (``|k| <= 2``): element ``x`` is
        ``a[x + k]``, and zero outside the array.
    """
    p = np.pad(a, [(2, 2) if ax == axis else (0, 0) for ax in range(a.ndim)])

    def at(k: int) -> np.ndarray:
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(2 + k, p.shape[axis] - 2 + k)
        return p[tuple(sl)]

    return at


def velocity_gradient(u: np.ndarray, fluid: np.ndarray, core: np.ndarray, h: float) -> np.ndarray:
    """Differentiate the velocity at the core voxels.

    Uses 4th-order central differences, and 2nd order where the 5-point stencil leaves the fluid.
    Where even the 3-point stencil leaves it, the derivative is 0, which reads as "no feature".

    Parameters
    ----------
    u : numpy.ndarray
        Velocity, ``(3, nx, ny, nz)``.
    fluid : numpy.ndarray
        Bool ``(nx, ny, nz)``, the voxels inside the mesh and outside the body.
    core : numpy.ndarray
        Bool ``(nx, ny, nz)``, the voxels to differentiate at.
    h : float
        The voxel size, in the length unit of the result.

    Returns
    -------
    numpy.ndarray
        Float32 ``(n_core, 3, 3)`` with ``G[n, i, j] = du_i / dx_j`` at the n-th core voxel.
    """
    G = np.empty((int(core.sum()), 3, 3), np.float32)
    for j in range(3):
        mask = taps(fluid, j)
        ok2 = mask(1) & mask(-1)
        ok4 = ok2 & mask(2) & mask(-2)
        for i in range(3):
            f = taps(u[i], j)
            d2 = (f(1) - f(-1)) / 2
            d4 = (8 * (f(1) - f(-1)) - (f(2) - f(-2))) / 12
            G[:, i, j] = (np.where(ok4, d4, np.where(ok2, d2, 0)) / h)[core]
    return G


def baseline_classes(x: np.ndarray, core: np.ndarray, contract: Contract, rules: LabelRules) -> np.ndarray:
    """Label a case with the rules, from the gradients of its own grid.

    Parameters
    ----------
    x : numpy.ndarray
        The unscaled inputs of a store, ``(6, nx, ny, nz)``.
    core : numpy.ndarray
        Bool ``(nx, ny, nz)``, the voxels where labels may be set.
    contract : Contract
        The input contract. Gives the voxel size, ``1 / n_per_H`` in units of H, so the gradients are
        dimensionless like the reference's.
    rules : LabelRules
        The label thresholds.

    Returns
    -------
    numpy.ndarray
        Uint8 ``(nx, ny, nz)``: 0 none, 1 vortex, 2 shear layer.
    """
    G = velocity_gradient(x[:3], x[4] < 1, core, 1 / contract.n_per_H)
    Q, omega = q_and_vorticity(G)
    vortex = (lambda2(G) < -rules.tau_lambda2) & (Q > 0)
    shear = (omega > rules.tau_omega) & ~vortex
    classes = np.zeros(core.shape, np.uint8)
    classes[core] = vortex + 2 * shear.astype(np.uint8)
    return classes
