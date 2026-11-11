"""
Verification: gradient-based ingredients on analytic linear velocity fields, for which the
mesh gradient is exact on any cell type, and their invariance to H and U_inf scaling.
"""

import numpy as np
import pytest
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from cfd_rgba_net.ingredients import add_ingredients, lambda2, q_and_vorticity
from cfd_rgba_net.preprocess import prepare_mesh

from helpers import box_mesh, interior, set_linear_velocity


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
    checks = add_ingredients(prep.mesh, 1.0)
    assert checks["gradient_of"] == ("cell U" if both else "point U")      # U_cell is present iff the mesh has cell U
    sel = interior(ug, 2) if both else slice(None)
    np.testing.assert_allclose(arr(prep.mesh, "Q_nd")[sel], Q, atol=1e-5)
    np.testing.assert_allclose(arr(prep.mesh, "omega_mag_nd")[sel], omega, atol=1e-5)
    np.testing.assert_allclose(arr(prep.mesh, "lambda2_nd")[sel], lam2, atol=1e-5)


def test_lambda2_is_independent_of_the_chunk_size():
    G = np.array([ROTATION, SHEAR, np.zeros((3, 3))], dtype=np.float32)
    np.testing.assert_array_equal(lambda2(G, n_chunk=1), lambda2(G))


def test_scaling_invariance(flow_cfg):
    """Scaling lengths by s (with H) and velocity by U_inf leaves every non-dimensional quantity unchanged."""
    def run(s, U_inf):
        ug = box_mesh()
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
