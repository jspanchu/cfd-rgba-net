import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkFeatureEdges, vtkTriangleFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkSphereSource
from vtkmodules.vtkIOGeometry import vtkSTLWriter

from cfd_rgba_net.config import Axis, Contract, axis_of
from cfd_rgba_net.geometry import CaseGeometry, ImageSpec, case_geometry, read_body, signed_distance
from cfd_rgba_net.preprocess import prepare_mesh

from helpers import box_mesh, set_linear_velocity


@pytest.mark.parametrize("direction,expected", [
    ((1, 0, 0), "+x"), ((0, 1, 0), "+y"), ((0, 0, -1), "-z"), ((0.99904, 0, -0.0438), "+x"),  # WindsorML yaw
])
def test_axis_of(direction, expected):
    assert str(axis_of(direction)) == expected


@pytest.mark.parametrize("direction", [(1, 1, 0), (0, 0, 0)])
def test_axis_of_rejects_off_axis(direction):
    with pytest.raises(ValueError):
        axis_of(direction)


CONTRACT = Contract("1.0", 10, ("u",))


def test_grid_fills_the_domain_from_the_ground():
    """Up along +x: the grid starts just above the ground and stays inside every face of the domain."""
    geo = CaseGeometry(body_bounds=(0.05, 0.35, -1.0, 0.0, -0.2, 0.2), ground=0.0, H=0.3, up=Axis(0, 1))
    domain = (0.0, 1.0, -3.0, 2.0, -0.75, 0.75)
    spec = geo.image_spec(CONTRACT, domain)
    b, dx = spec.bounds, spec.dx
    assert dx == pytest.approx(0.03)
    eps = 1e-6 * 0.3
    for a in range(3):
        assert domain[2 * a] < b[2 * a] and b[2 * a + 1] < domain[2 * a + 1]         # strictly inside
        assert domain[2 * a + 1] - b[2 * a + 1] < dx + eps                          # and fills it
    assert b[0] == pytest.approx(eps)                                               # ground side
    h = geo.height_above_ground(spec)
    assert h.shape == (spec.dims[0], 1, 1) and h.min() > 0


def test_grid_with_up_along_minus_axis():
    """Up along -z: the ground is the domain's top face and the grid is anchored there."""
    geo = CaseGeometry(body_bounds=(-1, 0, -0.2, 0.2, -0.35, -0.05), ground=0.0, H=0.3, up=Axis(2, -1))
    spec = geo.image_spec(CONTRACT, (-2.0, 3.0, -1.0, 1.0, -1.4, 0.0))
    assert spec.bounds[5] == pytest.approx(-1e-6 * 0.3)
    assert spec.bounds[4] > -1.4
    assert geo.height_above_ground(spec).min() > 0


def test_ground_auto_along_up_axis(flow_cfg):
    ug = box_mesh(lo=(-1.0, -1.0, 0.0), hi=(1.0, 1.0, 2.0))
    set_linear_velocity(ug, np.zeros((3, 3)), U0=(1, 0, 0))
    assert prepare_mesh(ug, flow_cfg(up=(0, 0, 1))).ground == pytest.approx(0.0)
    assert prepare_mesh(ug, flow_cfg(up=(0, -1, 0))).ground == pytest.approx(1.0)  # 'lowest' along -y = max y


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
    assert np.array_equal(inside, exact < 0) or np.abs(exact[inside != (exact < 0)]).max() < 2e-3


def test_body_scale_and_geometry(tmp_path):
    write_stl(vtkCubeSource(x_length=1000, y_length=400, z_length=300, center=(500, 0, 200)), tmp_path / "c.stl")
    body = read_body(tmp_path / "c.stl", scale=1e-3)
    geo = case_geometry(body, ground=0.0, reference_length=None, up=Axis(2, 1))
    assert geo.H == pytest.approx(0.3)
    assert geo.image_spec(CONTRACT, (-1.0, 3.0, -1.0, 1.0, 0.0, 1.0)).dx == pytest.approx(0.03)
    with pytest.raises(ValueError):
        case_geometry(body, ground=0.1, reference_length=None, up=Axis(2, 1))  # body below the ground


def test_stl_near_duplicate_vertices_are_merged(tmp_path):
    """Each triangle has its own copy of its vertices, jittered by ~1e-7 of the size (as in WindsorML)."""
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
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
    body = read_body(tmp_path / "split.stl")
    edges = vtkFeatureEdges(input_data=body, boundary_edges=True, feature_edges=False, manifold_edges=False,
                            non_manifold_edges=False)
    edges.Update()
    assert edges.GetOutput().GetNumberOfCells() == 0
    assert body.GetNumberOfPoints() == src.GetOutput().GetNumberOfPoints()


def test_stl_collapsed_triangles_are_dropped(tmp_path):
    """A sliver triangle whose points merge must vanish, not become a line (which crashed the distance)."""
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData
    src = vtkSphereSource(radius=0.5, center=(0, 0, 1), theta_resolution=32, phi_resolution=32)
    src.Update()
    sliver = vtkPolyData()
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray
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
    body = read_body(tmp_path / "sliver.stl")
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


def test_reference_probe_reads_u_inf_and_p_inf(flow_cfg):
    """U_inf = |U| and p_inf = p at the probe point; then u/U_inf there is a unit vector and Cp = 0."""
    import dataclasses
    from vtkmodules.util.numpy_support import vtk_to_numpy as v2n
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
    P = v2n(prep.mesh.GetPoints().GetData())
    i = np.argmin(np.linalg.norm(P - [0.25, 0.25, 0.75], axis=1))    # a mesh point near the probe
    Cp = v2n(prep.mesh.GetPointData().GetArray("Cp"))[i]
    assert abs(Cp - 2 * 30.0 * (P[i, 0] - 0.3) / (1.0 * prep.U_inf**2)) < 1e-6


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
