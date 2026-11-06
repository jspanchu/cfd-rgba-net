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
    """Middle eigenvalue of S^2 + W^2 for each (3, 3) velocity gradient in G. Shape (N,), float32."""
    out = np.empty(G.shape[0], dtype=np.float32)
    for i in range(0, G.shape[0], n_chunk):
        # float64: M subtracts terms of similar size and lambda2 is often close to 0
        g = G[i:i + n_chunk].astype(np.float64)
        gt = np.transpose(g, (0, 2, 1))
        S, W = 0.5 * (g + gt), 0.5 * (g - gt)
        out[i:i + n_chunk] = np.linalg.eigvalsh(S @ S + W @ W)[:, 1]
    return out


def q_and_vorticity(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Q = 0.5 (|W|^2 - |S|^2) and |omega| = |curl u| for each (3, 3) velocity gradient."""
    gt = np.transpose(G, (0, 2, 1))
    S, W = 0.5 * (G + gt), 0.5 * (G - gt)
    Q = 0.5 * (np.sum(W**2, axis=(1, 2)) - np.sum(S**2, axis=(1, 2)))
    w = np.stack([G[:, 2, 1] - G[:, 1, 2], G[:, 0, 2] - G[:, 2, 0], G[:, 1, 0] - G[:, 0, 1]], 1)
    return Q.astype(np.float32), np.linalg.norm(w, axis=1).astype(np.float32)


def add_ingredients(mesh: vtkUnstructuredGrid, H: float, dataset_vorticity: str | None = None, dataset_cp: str | None = None) -> dict:
    """
    Adds the INGREDIENTS as point data to mesh (prepared by preprocess.prepare_mesh), in place.
    The gradient is taken of the cell velocity when present (cell-centered, as OpenFOAM stores it),
    else of the point velocity. Optional dataset arrays are used only for sanity checks.
    Returns the checks.
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
