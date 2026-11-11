from dataclasses import replace

import numpy as np
import pytest
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkFeatureEdges, vtkTriangleFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkSphereSource
from vtkmodules.vtkIOGeometry import vtkSTLWriter

from cfd_rgba_net.config import Axis, Contract
from cfd_rgba_net.geometry import CaseGeometry, ImageSpec, case_geometry, read_body, signed_distance


CONTRACT = replace(Contract.load(), n_per_H=10)


def test_grid_fills_the_domain_from_the_ground():
    """Up along +x: the grid starts just above the ground and stays inside every face of the domain."""
    geo = CaseGeometry(body_bounds=(0.05, 0.35, -1.0, 0.0, -0.2, 0.2), ground=0.0, H=0.3, up=Axis(0, 1))
    domain = (0.0, 1.0, -3.0, 2.0, -0.75, 0.75)
    spec = geo.image_spec(CONTRACT, domain, floor_offset_H=1e-6, grid_inset_H=1e-6)
    b, dx = spec.bounds, spec.dx
    assert dx == pytest.approx(0.03)
    eps = 1e-6 * 0.3
    for a in range(3):
        assert domain[2 * a] < b[2 * a] and b[2 * a + 1] < domain[2 * a + 1]         # strictly inside
        assert domain[2 * a + 1] - b[2 * a + 1] < dx + eps                          # and fills it
    assert b[0] == pytest.approx(eps)                                               # ground side
    h = geo.height_above_ground(spec)
    assert h.shape == (spec.dims[0], 1, 1) and h.min() > 0


def test_grid_without_inset_or_offset_starts_on_the_domain_and_ground():
    geo = CaseGeometry(body_bounds=(0, 1, 0, 1, 0, 1), ground=0.0, H=0.3, up=Axis(2, 1))
    spec = geo.image_spec(CONTRACT, (0.0, 1.0, -1.0, 1.0, 0.0, 1.0))
    assert spec.origin == (0.0, -1.0, 0.0)


def test_image_spec_rejects_empty_domain():
    """A domain thinner than the ground offset leaves no room for a voxel."""
    geo = CaseGeometry(body_bounds=(0, 1, 0, 1, 0, 1), ground=0.0, H=1.0, up=Axis(2, 1))
    with pytest.raises(ValueError, match="empty domain along z"):
        geo.image_spec(CONTRACT, (0.0, 1.0, 0.0, 1.0, 0.0, 0.5), floor_offset_H=0.6)


def test_grid_with_up_along_minus_axis():
    """Up along -z: the ground is the domain's top face and the grid is anchored there."""
    geo = CaseGeometry(body_bounds=(-1, 0, -0.2, 0.2, -0.35, -0.05), ground=0.0, H=0.3, up=Axis(2, -1))
    spec = geo.image_spec(CONTRACT, (-2.0, 3.0, -1.0, 1.0, -1.4, 0.0), floor_offset_H=1e-6, grid_inset_H=1e-6)
    assert spec.bounds[5] == pytest.approx(-1e-6 * 0.3)
    assert spec.bounds[4] > -1.4
    assert geo.height_above_ground(spec).min() > 0


def write_stl(source, path):
    source.Update()
    tri = vtkTriangleFilter(input_data=source.GetOutput())
    tri.Update()
    vtkSTLWriter(file_name=str(path), input_data=tri.GetOutput()).Write()


def test_sphere_signed_distance(tmp_path):
    r = 0.5
    write_stl(vtkSphereSource(radius=r, center=(0, 0, 1), theta_resolution=128, phi_resolution=128), tmp_path / "s.stl")
    body = read_body(tmp_path / "s.stl")
    spec = ImageSpec(origin=(-1.0, -1.0, 0.0), dx=0.05, dims=(41, 41, 41))
    sdf, inside = signed_distance(body, spec)
    X, Y, Z = np.meshgrid(spec.axis(0), spec.axis(1), spec.axis(2), indexing="ij")
    exact = np.sqrt(X**2 + Y**2 + (Z - 1) ** 2) - r
    # tessellation error of a 128x128 sphere is < 1e-3 r
    np.testing.assert_allclose(sdf, exact, atol=2e-3)
    # the inside mask may differ from the analytic sphere only within the tessellation error of the surface
    mismatch = inside != (exact < 0)
    assert np.abs(exact[mismatch]).max(initial=0) < 2e-3


def test_signed_distance_requires_a_voxel_inside_the_body(tmp_path):
    write_stl(vtkSphereSource(radius=0.5, center=(0, 0, 1), theta_resolution=32, phi_resolution=32), tmp_path / "s.stl")
    body = read_body(tmp_path / "s.stl")
    away = ImageSpec(origin=(5.0, 5.0, 5.0), dx=0.1, dims=(5, 5, 5))     # a grid that misses the body
    with pytest.raises(ValueError, match="no voxel lies inside"):
        signed_distance(body, away)


@pytest.fixture
def cube_body(tmp_path):
    """A 1000 x 400 x 300 box (millimetres) with its bottom at z = 50, read at 1e-3 scale into metres."""
    write_stl(vtkCubeSource(x_length=1000, y_length=400, z_length=300, center=(500, 0, 200)), tmp_path / "c.stl")
    return read_body(tmp_path / "c.stl", scale=1e-3)


def test_read_body_applies_scale(cube_body):
    assert cube_body.GetBounds() == pytest.approx((0.0, 1.0, -0.2, 0.2, 0.05, 0.35))


def test_read_body_rejects_missing_and_empty_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_body(tmp_path / "missing.stl")
    (tmp_path / "empty.stl").write_text("solid empty\nendsolid empty\n")
    with pytest.raises(ValueError, match="no triangles"):
        read_body(tmp_path / "empty.stl")


def test_case_geometry_takes_h_from_body_height_or_override(cube_body):
    geo = case_geometry(cube_body, ground=0.05, reference_length=None, up=Axis(2, 1))
    assert geo.H == pytest.approx(0.3)
    assert geo.image_spec(CONTRACT, (-1.0, 3.0, -1.0, 1.0, 0.0, 1.0)).dx == pytest.approx(0.03)
    assert case_geometry(cube_body, ground=0.05, reference_length=0.6, up=Axis(2, 1)).H == 0.6


def test_case_geometry_rejects_body_below_ground(cube_body):
    with pytest.raises(ValueError, match="below the ground"):
        case_geometry(cube_body, ground=0.1, reference_length=None, up=Axis(2, 1))


def test_stl_near_duplicate_vertices_are_merged(tmp_path):
    """Each triangle has its own copy of its vertices, jittered by ~1e-7 of the size (as in WindsorML)."""
    src = vtkSphereSource(radius=0.5, theta_resolution=32, phi_resolution=32)
    src.Update()
    P = vtk_to_numpy(src.GetOutput().GetPoints().GetData()).astype(np.float64)
    T = vtk_to_numpy(src.GetOutput().GetPolys().GetConnectivityArray()).reshape(-1, 3)
    Q = P[T.ravel()] + np.random.default_rng(0).uniform(-1e-7, 1e-7, (T.size, 3))
    poly = vtkPolyData()
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(Q, deep=True))
    poly.SetPoints(pts)
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, T.size + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(T.size, dtype=np.int64), deep=True))
    poly.SetPolys(cells)
    vtkSTLWriter(file_name=str(tmp_path / "split.stl"), input_data=poly).Write()
    assert read_body(tmp_path / "split.stl").GetNumberOfPoints() > src.GetOutput().GetNumberOfPoints()  # open without a tolerance
    body = read_body(tmp_path / "split.stl", merge_tolerance=1e-5)
    edges = vtkFeatureEdges(input_data=body, boundary_edges=True, feature_edges=False, manifold_edges=False,
                            non_manifold_edges=False)
    edges.Update()
    assert edges.GetOutput().GetNumberOfCells() == 0
    assert body.GetNumberOfPoints() == src.GetOutput().GetNumberOfPoints()


def test_stl_collapsed_triangles_are_dropped(tmp_path):
    """A sliver triangle whose points merge must vanish, not become a line (which crashed the distance)."""
    src = vtkSphereSource(radius=0.5, center=(0, 0, 1), theta_resolution=32, phi_resolution=32)
    src.Update()
    sliver = vtkPolyData()
    pts = vtkPoints()
    for p in [(0.5, 0, 1), (0.5 + 1e-9, 0, 1), (0.5, 1e-9, 1.0 + 0.2)]:
        pts.InsertNextPoint(p)
    tri = vtkCellArray()
    tri.InsertNextCell(3, [0, 1, 2])
    sliver.SetPoints(pts)
    sliver.SetPolys(tri)
    app = vtkAppendPolyData()
    app.AddInputData(src.GetOutput())
    app.AddInputData(sliver)
    app.Update()
    vtkSTLWriter(file_name=str(tmp_path / "sliver.stl"), input_data=app.GetOutput()).Write()
    body = read_body(tmp_path / "sliver.stl", merge_tolerance=1e-5)
    assert body.GetNumberOfLines() == 0 and body.GetNumberOfVerts() == 0
    sdf, _ = signed_distance(body, ImageSpec(origin=(-1.0, -1.0, 0.0), dx=0.1, dims=(21, 21, 21)))
    assert np.isfinite(sdf).all()


def test_domain_above_ground_starts_at_domain():
    """A domain whose ground-side face is above the ground starts there, not at the ground."""
    up = CaseGeometry(body_bounds=(-1, 0, -0.2, 0.2, 0.05, 0.35), ground=0.0, H=0.3, up=Axis(2, 1))
    spec = up.image_spec(CONTRACT, (-2.0, 3.0, -1.0, 1.0, 0.5, 1.4), 1e-4)
    assert spec.origin[2] == pytest.approx(0.5, abs=1e-5) and spec.bounds[5] <= 1.4
    down = CaseGeometry(body_bounds=(-1, 0, -0.2, 0.2, -0.35, -0.05), ground=0.0, H=0.3, up=Axis(2, -1))
    spec = down.image_spec(CONTRACT, (-2.0, 3.0, -1.0, 1.0, -1.4, -0.5), 1e-4)
    assert spec.bounds[5] == pytest.approx(-0.5, abs=1e-5) and spec.origin[2] >= -1.4


def test_signed_distance_band_matches_exact(tmp_path):
    """Exact inside the band; outside it the distance transform is within a voxel of the exact distance."""
    r = 0.5
    write_stl(vtkSphereSource(radius=r, center=(0, 0, 1), theta_resolution=96, phi_resolution=96), tmp_path / "s.stl")
    body = read_body(tmp_path / "s.stl")
    spec = ImageSpec(origin=(-3.0, -2.0, 0.0), dx=0.05, dims=(121, 81, 61))
    exact, inside_e = signed_distance(body, spec)
    info: dict = {}
    fast, inside_f = signed_distance(body, spec, band=0.5, info=info)
    assert np.array_equal(inside_e, inside_f) and info["method"].startswith("exact in band")
    X, Y, Z = np.meshgrid(spec.axis(0), spec.axis(1), spec.axis(2), indexing="ij")
    in_band = (np.abs(X) < 0.975) & (np.abs(Y) < 0.975) & (np.abs(Z - 1) < 0.975)   # half a voxel inside the band
    np.testing.assert_allclose(fast[in_band], exact[in_band], atol=1e-6)
    err = np.abs(fast - exact)[~in_band]
    assert err.max() < 0.05 and np.median(err) < 0.01      # under a voxel (dx = 0.05); typically a fraction
