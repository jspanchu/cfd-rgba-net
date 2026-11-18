"""
Voxel-grid baseline: label rules v1 applied to finite-difference gradients of
the grid's own u, v, w.

It needs no training. The net must beat it (the baseline gate in the roadmap).
The same gradients give the net's inputs, ``grid_features``, and the baseline
thresholds those. ``grid_features`` runs in torch, on the GPU when there is one.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .config import LabelRules

# Channels of ``grid_features`` and of the stores' ``features`` array. Computed
# on the voxel grid, unlike the ingredients of the same names, which come from
# the native mesh and define the labels.
FEATURE_NAMES = ("lambda2", "Q", "omega_mag")


def default_device() -> torch.device:
    """Pick the device the grid quantities are computed on.

    Returns
    -------
    torch.device
        The first GPU when there is one, else the CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def taps(a: torch.Tensor, axis: int):
    """Give access to shifted copies of a tensor along one axis.

    Parameters
    ----------
    a : torch.Tensor
        The tensor. It is zero-padded by 2 voxels on both sides of ``axis``.
    axis : int
        The axis to shift along.

    Returns
    -------
    callable
        ``at(k)`` returns the view of ``a`` shifted by ``k`` voxels (``|k| <=
        2``): element ``x`` is ``a[x + k]``, and zero outside the tensor.
    """
    shape = list(a.shape)
    shape[axis] += 4
    p = a.new_zeros(shape)
    p.narrow(axis, 2, a.shape[axis]).copy_(a)

    def at(k: int) -> torch.Tensor:
        return p.narrow(axis, 2 + k, a.shape[axis])

    return at


def velocity_gradient(
    u: torch.Tensor, fluid: torch.Tensor, core: torch.Tensor, h: float
) -> torch.Tensor:
    """Differentiate the velocity at the core voxels.

    Uses 4th-order central differences, and 2nd order where the 5-point stencil
    leaves the fluid. Where even the 3-point stencil leaves it, the derivative
    is 0, which reads as "no feature".

    Parameters
    ----------
    u : torch.Tensor
        Float32 velocity, ``(3, nx, ny, nz)``.
    fluid : torch.Tensor
        Bool ``(nx, ny, nz)``, the voxels inside the mesh and outside the body.
    core : torch.Tensor
        Bool ``(nx, ny, nz)``, the voxels to differentiate at.
    h : float
        The voxel size, in the length unit of the result.

    Returns
    -------
    torch.Tensor
        Float32 ``(n_core, 3, 3)`` with ``G[n, i, j] = du_i / dx_j`` at the n-th
        core voxel, on ``u``'s device.
    """
    G = torch.empty(
        (int(core.sum()), 3, 3), dtype=torch.float32, device=u.device
    )
    for j in range(3):
        mask = taps(fluid, j)
        ok2 = mask(1) & mask(-1)
        ok4 = ok2 & mask(2) & mask(-2)
        for i in range(3):
            f = taps(u[i], j)
            d2 = (f(1) - f(-1)) / 2
            d4 = (8 * (f(1) - f(-1)) - (f(2) - f(-2))) / 12
            G[:, i, j] = (torch.where(ok4, d4, torch.where(ok2, d2, 0.0)) / h)[
                core
            ]
    return G


def q_and_vorticity(G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Q criterion and the vorticity magnitude for each velocity
    gradient.

    The same arithmetic as ``ingredients.q_and_vorticity``, in torch and in the
    dtype of ``G`` (float32).

    Parameters
    ----------
    G : torch.Tensor
        Velocity gradients, ``(N, 3, 3)``, with ``G[n, i, j] = du_i / dx_j``.

    Returns
    -------
    Q : torch.Tensor
        ``(N,)``: ``0.5 (|W|^2 - |S|^2)``, with Frobenius norms, ``S`` and ``W``
        the symmetric and antisymmetric parts of ``G``.
    omega : torch.Tensor
        ``(N,)``: the magnitude of ``curl u``.
    """
    gt = G.transpose(1, 2)
    S, W = 0.5 * (G + gt), 0.5 * (G - gt)
    Q = 0.5 * ((W**2).sum((1, 2)) - (S**2).sum((1, 2)))
    w = torch.stack(
        [
            G[:, 2, 1] - G[:, 1, 2],
            G[:, 0, 2] - G[:, 2, 0],
            G[:, 1, 0] - G[:, 0, 1],
        ],
        1,
    )
    return Q, torch.linalg.norm(w, dim=1)


def lambda2_closed_form(
    G: torch.Tensor, n_chunk: int = 4_000_000
) -> torch.Tensor:
    """Compute the lambda2 vortex criterion for each velocity gradient, from a
    closed-form eigenvalue.

    This differs on purpose from ``ingredients.lambda2``, which the labels are
    defined with: that one solves the eigenvalues with ``numpy.linalg.eigvalsh``
    and stays as it is, so the labels stay exactly as already defined. This one
    is for the grid features and the baseline only. It takes the eigenvalues of
    the symmetric 3 x 3 matrix ``M = S^2 + W^2`` from the trigonometric solution
    of its characteristic cubic (O. K. Smith, 1961), which vectorises on a GPU
    where a batched eigenvalue solver does not: about 0.2 s against 40 s for an
    AhmedML case. In float64 it agrees with ``eigvalsh`` to about 1e-12, and in
    float32 it gives the same value at 99.8 % of the voxels and one rounding
    step off at the rest.

    Parameters
    ----------
    G : torch.Tensor
        Velocity gradients, ``(N, 3, 3)``, with ``G[n, i, j] = du_i / dx_j``.
    n_chunk : int, optional
        Number of gradients processed at a time, which bounds the float64
        working memory.

    Returns
    -------
    torch.Tensor
        Float32 ``(N,)``: the middle eigenvalue of ``S^2 + W^2``, where ``S``
        and ``W`` are the symmetric and antisymmetric parts of ``G``. On ``G``'s
        device.
    """
    out = torch.empty(G.shape[0], dtype=torch.float32, device=G.device)
    eye = torch.eye(3, dtype=torch.float64, device=G.device)
    for s in range(0, G.shape[0], n_chunk):
        # float64: M subtracts terms of similar size and lambda2 is often close
        # to 0
        g = G[s : s + n_chunk].double()
        gt = g.transpose(1, 2)
        S, W = 0.5 * (g + gt), 0.5 * (g - gt)
        M = S @ S + W @ W
        diag = M.diagonal(dim1=1, dim2=2)
        q = diag.sum(1) / 3
        off = M[:, 0, 1] ** 2 + M[:, 0, 2] ** 2 + M[:, 1, 2] ** 2
        p = torch.sqrt((((diag - q[:, None]) ** 2).sum(1) + 2 * off) / 6)
        B = (M - q[:, None, None] * eye) / torch.where(p > 0, p, 1.0)[
            :, None, None
        ]
        phi = torch.arccos(torch.clamp(torch.linalg.det(B) / 2, -1, 1)) / 3
        # eigenvalues q + 2 p cos(phi + 2 pi k / 3): k = 0 is the largest, k = 1
        # the smallest; the trace gives the middle one
        largest, smallest = (
            q + 2 * p * torch.cos(phi),
            q + 2 * p * torch.cos(phi + 2 * math.pi / 3),
        )
        out[s : s + n_chunk] = (3 * q - largest - smallest).float()
    return out


def grid_features(
    u: np.ndarray,
    fluid: np.ndarray,
    h: float,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute the rules' three quantities at every voxel, from the grid's own
    velocity.

    Parameters
    ----------
    u : numpy.ndarray
        Velocity / U_inf, ``(3, nx, ny, nz)``.
    fluid : numpy.ndarray
        Bool ``(nx, ny, nz)``, the voxels inside the mesh and outside the body
        (``body_sdf > 0``).
    h : float
        The voxel size in units of H, ``1 / n_per_H``, so the quantities are
        dimensionless.
    device : torch.device or None, optional
        Where the arithmetic runs. ``None`` means ``default_device()``.

    Returns
    -------
    numpy.ndarray
        Float32 ``(3, nx, ny, nz)``: lambda2*, Q* and |omega|*, in the order of
        ``FEATURE_NAMES``. Voxels where no stencil fits, the body among them,
        get a zero gradient: lambda2* = Q* = |omega|* = 0.
    """
    device = device or default_device()
    ut = torch.from_numpy(np.ascontiguousarray(u, np.float32)).to(device)
    ft = torch.from_numpy(np.ascontiguousarray(fluid)).to(device)
    G = velocity_gradient(ut, ft, torch.ones_like(ft), h)
    del ut
    Q, omega = q_and_vorticity(G)
    return (
        torch.stack([lambda2_closed_form(G), Q, omega])
        .reshape(3, *fluid.shape)
        .cpu()
        .numpy()
    )


def baseline_classes(
    features: np.ndarray, core: np.ndarray, rules: LabelRules
) -> np.ndarray:
    """Label a case with the rules, from the features of its own grid.

    Parameters
    ----------
    features : numpy.ndarray
        The store's ``features``, ``(3, nx, ny, nz)``: lambda2*, Q* and
        |omega|*, in the order of ``FEATURE_NAMES``.
    core : numpy.ndarray
        Bool ``(nx, ny, nz)``, the voxels where labels may be set.
    rules : LabelRules
        The label thresholds.

    Returns
    -------
    numpy.ndarray
        Uint8 ``(nx, ny, nz)``: 0 none, 1 vortex, 2 shear layer.
    """
    lam2, Q, omega = features
    vortex = (lam2 < -rules.tau_lambda2) & (Q > 0)
    shear = (omega > rules.tau_omega) & ~vortex
    return (
        (vortex.astype(np.uint8) + 2 * shear.astype(np.uint8)) * core
    ).astype(np.uint8)
