"""
Label ingredients on the native mesh (teacher only, never model inputs):
    Q*, lambda2*, |omega|* from grad(u/U_inf) * H, cell-centered, distributed to the points.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkDataObject, vtkUnstructuredGrid
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData
from vtkmodules.vtkFiltersGeneral import vtkGradientFilter

from .preprocess import CP, U_CELL, U_ND

INGREDIENTS = ("Q_nd", "lambda2_nd", "omega_mag_nd")


def lambda2(G: np.ndarray, n_chunk: int = 2_000_000) -> np.ndarray:
    """Compute the lambda2 vortex criterion for each velocity gradient.

    Parameters
    ----------
    G : numpy.ndarray
        Velocity gradients, shape ``(N, 3, 3)``, with ``G[n, i, j] = du_i / dx_j``.
    n_chunk : int, optional
        Number of gradients processed at a time, which bounds the memory of the eigenvalue solve.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(N,)``: the middle eigenvalue of ``S^2 + W^2``, where ``S`` and ``W``
        are the symmetric and antisymmetric parts of ``G``.
    """
    out = np.empty(G.shape[0], dtype=np.float32)
    for i in range(0, G.shape[0], n_chunk):
        # float64: M subtracts terms of similar size and lambda2 is often close to 0
        g = G[i:i + n_chunk].astype(np.float64)
        gt = np.transpose(g, (0, 2, 1))
        S, W = 0.5 * (g + gt), 0.5 * (g - gt)
        out[i:i + n_chunk] = np.linalg.eigvalsh(S @ S + W @ W)[:, 1]
    return out


def q_and_vorticity(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Q criterion and the vorticity magnitude for each velocity gradient.

    The arithmetic runs in the dtype of ``G``, which is float32 in the pipeline. Unlike ``lambda2``, it
    is not promoted to float64, although ``Q`` also subtracts terms of similar size. So ``Q`` carries
    float32 round-off where ``|W|^2`` and ``|S|^2`` nearly cancel, as in pure shear.

    Parameters
    ----------
    G : numpy.ndarray
        Velocity gradients, shape ``(N, 3, 3)``, with ``G[n, i, j] = du_i / dx_j``.

    Returns
    -------
    Q : numpy.ndarray
        Float32 array of shape ``(N,)``: ``0.5 (|W|^2 - |S|^2)``, with Frobenius norms.
    omega : numpy.ndarray
        Float32 array of shape ``(N,)``: the magnitude of ``curl u``.
    """
    gt = np.transpose(G, (0, 2, 1))
    S, W = 0.5 * (G + gt), 0.5 * (G - gt)
    Q = 0.5 * (np.sum(W**2, axis=(1, 2)) - np.sum(S**2, axis=(1, 2)))
    w = np.stack([G[:, 2, 1] - G[:, 1, 2], G[:, 0, 2] - G[:, 2, 0], G[:, 1, 0] - G[:, 0, 1]], 1)
    return Q.astype(np.float32), np.linalg.norm(w, axis=1).astype(np.float32)


def add_ingredients(
    mesh: vtkUnstructuredGrid,
    H: float,
    dataset_vorticity: str | None = None,
    dataset_cp: str | None = None,
) -> dict:
    """Add the label ingredients to a prepared mesh.

    The gradient is taken of the cell velocity (``U_cell``, cell-centered, as OpenFOAM stores it) when the
    mesh has it, else of the point velocity (``U_nd``). It is scaled by ``H`` to make it dimensionless.
    Gradients that are not finite (degenerate cells) are set to zero and counted in the checks, so they
    read as "no feature". Cell values are averaged to the points.

    Parameters
    ----------
    mesh : vtkUnstructuredGrid
        The mesh from ``preprocess.prepare_mesh``. Modified in place: ``INGREDIENTS`` are added to its
        point data.
    H : float
        Reference length, in mesh units.
    dataset_vorticity : str or None, optional
        Name of the dataset's own vorticity array in the cell data. When given, and the gradient was
        taken of the cell velocity, the checks compare it with ``|omega|*``.
    dataset_cp : str or None, optional
        Name of the dataset's own pressure-coefficient array in the point data. When given and present,
        the checks report its largest difference from ``Cp``. A cell-only mesh never has it there.

    Returns
    -------
    dict
        The checks:

        - ``gradient_of``: ``"cell U"`` or ``"point U"``;
        - ``n_nonfinite_gradient``: number of gradients set to zero;
        - ``omega_vs_dataset``: ``array``, ``spearman_r`` and ``note``, from a random 2M-cell subsample
          above the median. Only when ``dataset_vorticity`` applies;
        - ``Cp_vs_dataset_maxabs``: only when ``dataset_cp`` applies.
    """
    use_cell = mesh.GetCellData().HasArray(U_CELL)
    grad = vtkGradientFilter(input_data=mesh, result_array_name="gradU", compute_divergence=False,
                             compute_q_criterion=False, compute_vorticity=False)
    if use_cell:
        grad.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, U_CELL)
    else:
        grad.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, U_ND)
    grad.Update()
    gout = grad.GetOutput()
    gdata = gout.GetCellData() if use_cell else gout.GetPointData()
    G = (vtk_to_numpy(gdata.GetArray("gradU")).reshape(-1, 3, 3) * H).astype(np.float32)
    bad = ~np.isfinite(G).all(axis=(1, 2))
    G[bad] = 0.0
    Q, omega = q_and_vorticity(G)
    lam2 = lambda2(G)
    del G
    checks: dict = {"gradient_of": "cell U" if use_cell else "point U", "n_nonfinite_gradient": int(bad.sum())}

    if dataset_vorticity and mesh.GetCellData().HasArray(dataset_vorticity) and use_cell:
        ref = np.linalg.norm(vtk_to_numpy(mesh.GetCellData().GetArray(dataset_vorticity)), axis=1)
        # the dataset array is dimensional; compare the pattern (rank) and the scale (median ratio)
        rng = np.random.default_rng(0)
        idx = rng.choice(ref.size, min(2_000_000, ref.size), replace=False)
        sel = idx[ref[idx] > np.median(ref)]
        checks["omega_vs_dataset"] = dict(
            array=dataset_vorticity,
            spearman_r=float(spearmanr(omega[sel], ref[sel]).statistic),
            note="per cell, 2M-cell random subsample above the median; scale differs by H/U_inf",
        )

    if use_cell:
        tmp = vtkUnstructuredGrid()
        tmp.ShallowCopy(mesh)
        tmp.GetPointData().Initialize()
        tmp.GetCellData().Initialize()
        for name, a in (("Q_nd", Q), ("lambda2_nd", lam2), ("omega_mag_nd", omega)):
            v = numpy_to_vtk(a, deep=True)
            v.SetName(name)
            tmp.GetCellData().AddArray(v)
        c2p = vtkCellDataToPointData(input_data=tmp)
        c2p.Update()
        point = {n: vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray(n)).astype(np.float32)
                 for n in ("Q_nd", "lambda2_nd", "omega_mag_nd")}
    else:
        point = {"Q_nd": Q, "lambda2_nd": lam2, "omega_mag_nd": omega}

    if dataset_cp and mesh.GetPointData().HasArray(dataset_cp):
        Cp = vtk_to_numpy(mesh.GetPointData().GetArray(CP))
        checks["Cp_vs_dataset_maxabs"] = float(np.abs(Cp - vtk_to_numpy(mesh.GetPointData().GetArray(dataset_cp))).max())
    for name, a in point.items():
        v = numpy_to_vtk(np.ascontiguousarray(a), deep=True)
        v.SetName(name)
        mesh.GetPointData().AddArray(v)
    return checks
