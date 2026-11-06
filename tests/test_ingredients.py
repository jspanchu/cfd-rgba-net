"""
Verification: gradient-based ingredients on analytic linear velocity fields, for which the
mesh gradient is exact on any cell type, and their invariance to H and U_inf scaling.
"""

import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy

from cfd_rgba_net.config import VectorField
from cfd_rgba_net.ingredients import add_ingredients, lambda2, q_and_vorticity
from cfd_rgba_net.preprocess import prepare_mesh

from helpers import box_mesh, set_linear_velocity


def arr(ug, name):
    return vtk_to_numpy(ug.GetPointData().GetArray(name))


W = 0.7     # rotation rate
GAMMA = 1.3  # shear rate
ROTATION = [[0, -W, 0], [W, 0, 0], [0, 0, 0]]   # u = (-W y, W x, 0)
SHEAR = [[0, GAMMA, 0], [0, 0, 0], [0, 0, 0]]   # u = (GAMMA y, 0, 0)


def test_closed_forms():
    G = np.array([ROTATION, SHEAR], dtype=np.float32)
    Q, om = q_and_vorticity(G)
    l2 = lambda2(G)
    np.testing.assert_allclose(Q, [W**2, 0.0], atol=1e-6)
    np.testing.assert_allclose(om, [2 * W, GAMMA], atol=1e-6)
    np.testing.assert_allclose(l2, [-W**2, 0.0], atol=1e-6)


@pytest.mark.parametrize("tets", [True, False], ids=["tets", "hexes"])
@pytest.mark.parametrize("both", [True, False], ids=["cell+point U", "point U only"])
@pytest.mark.parametrize("field,Q,omega,lam2", [
    (ROTATION, W**2, 2 * W, -W**2),
    (SHEAR, 0.0, GAMMA, 0.0),
], ids=["solid-body rotation", "pure shear"])
def test_linear_fields_exact(flow_cfg, tets, both, field, Q, omega, lam2):
    """
    Point-U gradients are exact everywhere. Cell-U gradients (vtkGradientFilter on cell data, the
    path for OpenFOAM data) are exact except in the cell layer touching the mesh boundary, so after
    averaging to the points they are exact at points 2 cells or more from the boundary. Near walls
    that layer is far thinner than a voxel and is removed by the label rules' wall erosion.
    """
    ug = box_mesh(tets=tets)
    set_linear_velocity(ug, field, both=both)
    prep = prepare_mesh(ug, flow_cfg())
    add_ingredients(prep.mesh, 1.0)
    sel = interior(ug, 2) if both else slice(None)
    np.testing.assert_allclose(arr(prep.mesh, "Q_nd")[sel], Q, atol=1e-5)
    np.testing.assert_allclose(arr(prep.mesh, "omega_mag_nd")[sel], omega, atol=1e-5)
    np.testing.assert_allclose(arr(prep.mesh, "lambda2_nd")[sel], lam2, atol=1e-5)


def interior(ug, layers, lo=(-1.0, -1.0, 0.0), hi=(1.0, 1.0, 2.0), h=0.25):
    """Points at least `layers` cells (of size h) from every face of box_mesh's default box."""
    P = vtk_to_numpy(ug.GetPoints().GetData())
    d = layers * h - 1e-9
    return np.all((P >= np.array(lo) + d) & (P <= np.array(hi) - d), axis=1)


def test_cell_gradient_error_confined_to_boundary_layer():
    """Documents the vtkGradientFilter cell-data behaviour the test above works around."""
    from vtkmodules.vtkCommonDataModel import vtkDataObject
    from vtkmodules.vtkFiltersGeneral import vtkGradientFilter
    from helpers import cell_centers
    ug = box_mesh(tets=False)
    set_linear_velocity(ug, ROTATION)
    g = vtkGradientFilter(input_data=ug, result_array_name="g")
    g.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, "UMean")
    g.Update()
    J = vtk_to_numpy(g.GetOutput().GetCellData().GetArray("g")).reshape(-1, 3, 3)
    err = np.abs(J - np.array(ROTATION)).max(axis=(1, 2))
    C = cell_centers(ug)
    first_layer = np.any((C < np.array([-1, -1, 0]) + 0.25) | (C > np.array([1, 1, 2]) - 0.25), axis=1)
    assert err[~first_layer].max() < 1e-6
    assert err[first_layer].max() > 0.1


def test_scaling_invariance(flow_cfg):
    """Scaling lengths by s (with H) and velocity by U_inf leaves every non-dimensional quantity unchanged."""
    def run(s, U_inf):
        ug = box_mesh()
        from vtkmodules.util.numpy_support import numpy_to_vtk
        P = vtk_to_numpy(ug.GetPoints().GetData()) * s
        ug.GetPoints().SetData(numpy_to_vtk(P, deep=True))
        # same non-dimensional field: u/U_inf = (1 + 0.2 y/H, 0.1 z/H, 0), Cp = 0.3 x/H
        G = np.array([[0, 0.2, 0], [0, 0, 0.1], [0, 0, 0]]) * U_inf / s
        set_linear_velocity(ug, G, U0=(U_inf, 0, 0), p=lambda X: 0.5 * 1.2 * U_inf**2 * 0.3 * X[:, 0] / s)
        prep = prepare_mesh(ug, flow_cfg(U_inf=U_inf, rho=1.2))
        add_ingredients(prep.mesh, s)
        return {n: arr(prep.mesh, n).copy() for n in ("U_nd", "Cp", "Q_nd", "lambda2_nd", "omega_mag_nd")}

    a, b = run(1.0, 1.0), run(10.0, 40.0)
    for n in a:
        np.testing.assert_allclose(a[n], b[n], rtol=1e-4, atol=1e-5, err_msg=n)


def test_velocity_components_and_cell_only(flow_cfg):
    """Velocity as three scalar cell arrays (WindsorML layout) gives the same point U as a point vector."""
    ug = box_mesh()
    set_linear_velocity(ug, ROTATION, U0=(1, 0, 0))
    U_point = arr(ug, "UMean").copy()
    from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
    from helpers import add, cell_centers
    cells = vtkUnstructuredGrid()
    cells.ShallowCopy(ug)
    cells.GetPointData().Initialize()
    cells.GetCellData().Initialize()
    C = cell_centers(ug)
    Uc = np.array([1, 0, 0]) + C @ np.array(ROTATION, dtype=float).T
    for i, c in enumerate("xyz"):
        add(cells, f"velocity{c}avg", Uc[:, i], "cell")
    add(cells, "pressureavg", np.zeros(len(C)), "cell")
    cfg = flow_cfg(velocity=(VectorField(components=("velocityxavg", "velocityyavg", "velocityzavg")),))
    cfg = cfg.__class__(**{**cfg.__dict__, "pressure": ("pressureavg",)})
    prep = prepare_mesh(cells, cfg)
    # cell-to-point averaging of a linear field is exact in the interior, not on the boundary
    inner = interior(ug, 1)
    np.testing.assert_allclose(arr(prep.mesh, "U_nd")[inner], U_point[inner], atol=1e-5)
    add_ingredients(prep.mesh, 1.0)
    np.testing.assert_allclose(arr(prep.mesh, "Q_nd")[interior(ug, 2)], W**2, atol=1e-5)
