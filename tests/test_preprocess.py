"""Verification: reading the volume, selecting and scaling the fields, probing, and resampling to the voxel grid."""

import dataclasses

import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

from cfd_rgba_net.config import VectorField
from cfd_rgba_net.geometry import ImageSpec
from cfd_rgba_net.preprocess import U_CELL, U_ND, prepare_mesh, probe_point, read_volume, resample

from helpers import add, box_mesh, cell_centers, interior, set_linear_velocity


def test_read_volume_rejects_missing_and_empty_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_volume(tmp_path / "missing.vtu")
    empty = tmp_path / "empty.vtu"
    vtkXMLUnstructuredGridWriter(file_name=str(empty), input_data=vtkUnstructuredGrid()).Write()
    with pytest.raises(ValueError, match="no points"):
        read_volume(empty)


@pytest.mark.parametrize("source", ["point", "cell"])
def test_field_source_must_be_present_in_the_mesh(flow_cfg, source):
    """No fallback: a mesh with only the other kind of data fails, naming the requested source."""
    ug = box_mesh()
    set_linear_velocity(ug, np.zeros((3, 3)), U0=(1, 0, 0))
    (ug.GetPointData() if source == "point" else ug.GetCellData()).Initialize()    # drop the requested kind only
    with pytest.raises(KeyError, match=f"field_source is '{source}'"):
        prepare_mesh(ug, flow_cfg(field_source=source))


def test_cell_velocity_components_are_averaged_to_points(flow_cfg):
    """Velocity as three scalar cell arrays (WindsorML layout) gives the same point U as a point vector."""
    G = np.array([[0, -0.7, 0], [0.7, 0, 0], [0, 0, 0]])    # u = (-0.7 y, 0.7 x, 0)
    ug = box_mesh()
    set_linear_velocity(ug, G, U0=(1, 0, 0))
    U_point = vtk_to_numpy(ug.GetPointData().GetArray("UMean")).copy()
    cells = vtkUnstructuredGrid()
    cells.ShallowCopy(ug)
    cells.GetPointData().Initialize()
    cells.GetCellData().Initialize()
    C = cell_centers(ug)
    Uc = np.array([1, 0, 0]) + C @ G.T
    for i, c in enumerate("xyz"):
        add(cells, f"velocity{c}avg", Uc[:, i], "cell")
    add(cells, "pressureavg", np.zeros(len(C)), "cell")
    cfg = flow_cfg(velocity=(VectorField(components=("velocityxavg", "velocityyavg", "velocityzavg")),),
                   field_source="cell")
    prep = prepare_mesh(cells, dataclasses.replace(cfg, pressure=("pressureavg",)))
    # cell-to-point averaging of a linear field is exact in the interior, not on the boundary
    inner = interior(ug, 1)
    np.testing.assert_allclose(vtk_to_numpy(prep.mesh.GetPointData().GetArray(U_ND))[inner], U_point[inner], atol=1e-5)


@pytest.mark.parametrize("has_cell_velocity", [True, False])
def test_cell_velocity_is_kept_for_gradients_when_the_mesh_has_it(flow_cfg, has_cell_velocity):
    """Point fields are used for U and Cp either way; U_cell appears only if the mesh holds cell velocity."""
    ug = box_mesh()
    set_linear_velocity(ug, np.zeros((3, 3)), U0=(1, 0, 0), both=has_cell_velocity)
    prep = prepare_mesh(ug, flow_cfg(field_source="point"))
    assert bool(prep.mesh.GetCellData().HasArray(U_CELL)) == has_cell_velocity


@pytest.mark.parametrize("flow,match", [({"U_inf": 0.0}, "U_inf"), ({"rho": 0.0}, "rho")])
def test_non_positive_reference_speed_or_density_is_rejected(flow_cfg, flow, match):
    ug = box_mesh()
    set_linear_velocity(ug, np.zeros((3, 3)), U0=(1, 0, 0))
    with pytest.raises(ValueError, match=match):
        prepare_mesh(ug, flow_cfg(**flow))


def test_ground_auto_along_up_axis(flow_cfg):
    ug = box_mesh(lo=(-1.0, -1.0, 0.0), hi=(1.0, 1.0, 2.0))
    set_linear_velocity(ug, np.zeros((3, 3)), U0=(1, 0, 0))
    assert prepare_mesh(ug, flow_cfg(up=(0, 0, 1))).ground == pytest.approx(0.0)
    assert prepare_mesh(ug, flow_cfg(up=(0, -1, 0))).ground == pytest.approx(1.0)  # 'lowest' along -y = max y


def test_reference_probe_reads_u_inf_and_p_inf(flow_cfg):
    """U_inf = |U| and p_inf = p at the probe point; then u/U_inf there is a unit vector and Cp = 0."""
    ug = box_mesh()
    G = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    set_linear_velocity(ug, G, U0=(40.0, 0.0, 1.5), p=lambda X: 1000.0 + 30.0 * X[:, 0])
    base = flow_cfg()
    cfg = dataclasses.replace(base, flow=dataclasses.replace(base.flow, U_inf=None, p_inf=None,
                                                             reference_probe=(0.3, 0.2, 0.7)))
    prep = prepare_mesh(ug, cfg)
    U_ref = np.array([40.0 + 0.5 * 0.2, 0.0, 1.5])
    assert prep.U_inf == pytest.approx(np.linalg.norm(U_ref), rel=1e-6)
    assert prep.p_inf == pytest.approx(1000.0 + 30.0 * 0.3, rel=1e-6)
    P = vtk_to_numpy(prep.mesh.GetPoints().GetData())
    i = np.argmin(np.linalg.norm(P - [0.25, 0.25, 0.75], axis=1))    # a mesh point near the probe
    Cp = vtk_to_numpy(prep.mesh.GetPointData().GetArray("Cp"))[i]
    assert abs(Cp - 2 * 30.0 * (P[i, 0] - 0.3) / (1.0 * prep.U_inf**2)) < 1e-6


def test_probe_point_outside_the_mesh_is_rejected():
    ug = box_mesh()
    n = ug.GetNumberOfPoints()
    with pytest.raises(ValueError, match="not inside any mesh cell"):
        probe_point(ug, (5.0, 0.0, 0.5), np.zeros((n, 3)), np.zeros(n))


def test_resample_indexes_x_y_z_and_flags_voxels_outside_the_mesh(flow_cfg):
    """u = (x, z, 0) on a box mesh x in [-1, 1]: the grid sticks out in x, and its dims differ per axis."""
    ug = box_mesh(lo=(-1.0, -1.0, 0.0), hi=(1.0, 1.0, 2.0))
    set_linear_velocity(ug, np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 0, 0]]))
    prep = prepare_mesh(ug, flow_cfg(U_inf=1.0))
    spec = ImageSpec(origin=(-1.25, -0.3, 0.6), dx=0.5, dims=(6, 2, 2))     # x = -1.25 and 1.25 are outside
    image = resample(prep.mesh, spec, [U_ND])
    X, _, Z = np.meshgrid(spec.axis(0), spec.axis(1), spec.axis(2), indexing="ij")
    valid = image["valid"]
    assert valid.shape == spec.dims
    assert valid[:, 0, 0].tolist() == [False, True, True, True, True, False]
    np.testing.assert_allclose(image[U_ND][0][valid], X[valid], atol=1e-6)
    np.testing.assert_allclose(image[U_ND][1][valid], Z[valid], atol=1e-6)
