"""
End to end on a tiny synthetic case: a sphere inside a hexahedral mesh that also fills the sphere
(immersed-boundary layout), uniform flow plus a Lamb-Oseen vortex along x through y = 0, z = 0.6.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import zarr
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOGeometry import vtkSTLWriter
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

from cfd_rgba_net.build_case import build_case
from cfd_rgba_net.config import CaseConfig, Contract, LabelRules

from helpers import add, box_mesh, cell_centers, points

DATASET_YAML = """
dataset: Synthetic
files:
  volume: run_{run}/volume_{run}.vtu
  body: run_{run}/body_{run}.stl
fields:
  velocity: [UMean]
  pressure: [pMean]
field_source: cell
flow:
  U_inf: 10.0
  p_inf: 0.0
  rho: 1.0
  nu: 1.0e-5
  up_direction: [0, 0, 1]
  ground: auto
floor_offset_H: 1.0e-6
grid_inset_H: 1.0e-6
body:
  scale: 1.0
  reference_length: height
"""


RC = 0.1        # vortex core radius
GAMMA = 15.7    # circulation: peak vorticity* = GAMMA / (pi RC^2) * H / U_inf ~ 20 with H = 0.4


def velocity(X):
    U = np.zeros_like(X)
    U[:, 0] = 10.0
    dy, dz = X[:, 1], X[:, 2] - 0.6
    r2 = np.maximum(dy**2 + dz**2, 1e-12)
    ut_over_r = GAMMA / (2 * np.pi * r2) * (1 - np.exp(-r2 / RC**2))
    U[:, 1] = -ut_over_r * dz
    U[:, 2] = ut_over_r * dy
    return U


def make_case(root):
    d = root / "run_1"
    d.mkdir(parents=True)
    sphere = vtkSphereSource(radius=0.2, center=(0, 0, 0.6), theta_resolution=64, phi_resolution=64)
    sphere.Update()
    tri = vtkTriangleFilter(input_data=sphere.GetOutput())
    tri.Update()
    vtkSTLWriter(file_name=str(d / "body_1.stl"), input_data=tri.GetOutput()).Write()
    ug = box_mesh(n=(81, 41, 41), lo=(-1.0, -0.75, 0.0), hi=(3.0, 0.75, 1.5), tets=False)
    P, C = points(ug), cell_centers(ug)
    add(ug, "UMean", velocity(P))
    add(ug, "UMean", velocity(C), "cell")
    add(ug, "pMean", np.zeros(len(P)))
    add(ug, "pMean", np.zeros(len(C)), "cell")
    vtkXMLUnstructuredGridWriter(file_name=str(d / "volume_1.vtu"), input_data=ug).Write()
    (root / "synthetic.yaml").write_text(DATASET_YAML)


CONTRACT = replace(Contract.load(), n_per_H=16)


def test_build_case_fails_without_body_voxels(tmp_path):
    """A domain that misses the body is an error, and no store is written."""
    make_case(tmp_path)
    yaml = (tmp_path / "synthetic.yaml")
    yaml.write_text(DATASET_YAML + "domain: [1.5, 3.0, -0.75, 0.75, 0.0, 1.5]\n")
    cfg = CaseConfig.load(yaml)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no voxel lies inside the body"):
        build_case(cfg, 1, tmp_path, out, CONTRACT, LabelRules.load(), log=lambda m: None)
    assert not (out / "synthetic_run1.zarr").exists()


@dataclass
class BuiltCase:
    out: Path
    meta: dict
    store: zarr.Group


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """The synthetic case, built once for the tests that read the result."""
    root = tmp_path_factory.mktemp("case")
    make_case(root)
    out = root / "out"
    cfg = CaseConfig.load(root / "synthetic.yaml")
    meta = build_case(cfg, 1, root, out, CONTRACT, LabelRules.load(), save_vti=True, log=lambda m: None)
    return BuiltCase(out, meta, zarr.open_group(out / "synthetic_run1.zarr", mode="r"))


def test_store_layout_provenance_and_grid(case):
    store, m = case.store, dict(case.store.attrs)
    assert not (case.out / "synthetic_run1.zarr.tmp").exists()
    assert set(store.array_keys()) == {"features", "labels", "body_sdf", "height"}
    assert store["features"].shape[0] == 3 and store["features"].dtype == np.float32
    assert store["body_sdf"].shape == store["height"].shape == store["labels"].shape[1:]
    assert m["class_names"] == ["vortex", "shear_layer"]
    assert set(store["ingredients"].array_keys()) == {"Q_nd", "lambda2_nd", "omega_mag_nd", "core"}
    assert abs(m["geometry"]["H"] - 0.4) < 1e-3                  # the sphere's diameter
    assert len(m["provenance"]["inputs"]["volume"]["sha256"]) == 64
    for f in ("features", "ingredients", "labels"):               # save_vti=True
        assert (case.out / f"synthetic_run1_{f}.vti").exists()
    # the mesh is x -1..3, z 0..1.5: the grid starts just inside its faces
    g = case.meta["grid"]
    assert -1.0 < g["origin"][0] < -1.0 + 1e-3 and 0.0 < g["origin"][2] < 1e-3


def test_body_sdf_is_negative_exactly_in_the_body(case):
    sdf = case.store["body_sdf"][...]
    # H = sphere diameter; sdf/H at the sphere center = -radius/H = -0.5
    assert -0.51 < sdf.min() < -0.45
    voxels = case.meta["checks"]["voxels"]
    # body voxels are inside the mesh here (immersed boundary), but the STL mask removes them from the fluid
    assert voxels["n_in_mesh_and_body"] > 0 and voxels["geometry"]["n_sdf_neg_not_body"] == 0


def test_height_is_a_function_of_the_up_axis_only(case):
    """Ground at z = 0 and H = 0.4: the height is z / H on the layers."""
    h = case.store["height"][...]
    g = case.meta["grid"]
    z = g["origin"][2] + float(g["dx"]) * np.arange(h.shape[2])
    np.testing.assert_allclose(h[0, 0], z / case.meta["geometry"]["H"], atol=1e-5)
    assert (h == h[:1, :1, :]).all()


def test_vortex_labels_hug_the_vortex_axis(case):
    m = dict(case.store.attrs)
    vortex = case.store["labels"][...][m["class_names"].index("vortex")]
    o, dx = m["grid"]["origin"], float(m["grid"]["dx"])
    i, j, k = np.nonzero(vortex)
    r = np.hypot(o[1] + dx * j, o[2] + dx * k - 0.6)
    assert vortex.sum() > 0 and (r < 2 * RC).all()
    # the axis behind the body is labelled
    xs = o[0] + dx * np.arange(vortex.shape[0])
    jy, kz = round(-o[1] / dx), round((0.6 - o[2]) / dx)
    assert vortex[(xs > 0.5) & (xs < 2.5), jy, kz].mean() > 0.9
