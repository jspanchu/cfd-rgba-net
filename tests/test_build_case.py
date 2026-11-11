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

from cfd_rgba_net.build_case import build_case, read_geo_parameters
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


def test_build_case_refuses_a_grid_over_max_voxels(tmp_path):
    make_case(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="> 1,000"):
        build_case(CaseConfig.load(tmp_path / "synthetic.yaml"), 1, tmp_path, out, CONTRACT, LabelRules.load(),
                   log=lambda m: None, max_voxels=1000)
    assert not (out / "synthetic_run1.zarr").exists()


@pytest.mark.parametrize("contract,nu,match", [
    (replace(CONTRACT, channels=("u",)), 1.0e-5, "channels"),
    (CONTRACT, 0.0, "flow.nu"),
])
def test_build_case_rejects_inconsistent_settings_before_reading_any_file(tmp_path, contract, nu, match):
    """The data directory is empty: a ValueError, not a FileNotFoundError, shows nothing was read."""
    (tmp_path / "synthetic.yaml").write_text(DATASET_YAML)
    cfg = CaseConfig.load(tmp_path / "synthetic.yaml")
    cfg = replace(cfg, flow=replace(cfg.flow, nu=nu))
    with pytest.raises(ValueError, match=match):
        build_case(cfg, 1, tmp_path, tmp_path / "out", contract, LabelRules.load(), log=lambda m: None)


def test_read_geo_parameters(tmp_path):
    f = tmp_path / "geo.csv"
    f.write_text("run, length ,width\n1, 1.5, 2\n")
    assert read_geo_parameters(f) == {"run": 1.0, "length": 1.5, "width": 2.0}
    assert read_geo_parameters(tmp_path / "missing.csv") == {}


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


def test_store_layout_and_metadata(case):
    store, m = case.store, dict(case.store.attrs)
    assert not (case.out / "synthetic_run1.zarr.tmp").exists()
    assert set(store.array_keys()) == {"inputs", "labels"}       # no 0/1 body or fluid mask: the body is b > 1
    assert store["inputs"].shape[0] == 6 and store["inputs"].dtype == np.float32
    assert store["inputs"].chunks == (6, 64, 64, 64)
    assert m["channel_names"] == ["u", "v", "w", "Cp", "body_proximity", "ground_proximity"]
    assert m["class_names"] == ["vortex", "shear_layer"]
    assert set(store["ingredients"].array_keys()) == {"Q_nd", "lambda2_nd", "omega_mag_nd", "core"}
    assert abs(m["geometry"]["H"] - 0.4) < 1e-3                  # the sphere's diameter


def test_provenance_records_versions_and_input_hashes(case):
    prov = dict(case.store.attrs)["provenance"]
    assert prov["contract_version"] == "1.0" and prov["label_rules_version"] == "1"
    assert len(prov["inputs"]["volume"]["sha256"]) == 64


def test_vti_files_are_written_on_request(case):
    for f in ("inputs", "ingredients", "labels"):
        assert (case.out / f"synthetic_run1_{f}.vti").exists()


def test_grid_fills_the_mesh_domain_just_inside_its_faces(case):
    """The mesh is x -1..3, y -0.75..0.75, z 0..1.5."""
    g = case.meta["grid"]
    assert -1.0 < g["origin"][0] < -1.0 + 1e-3 and 0.0 < g["origin"][2] < 1e-3


def test_body_proximity_marks_the_body_and_inputs_vanish_outside_the_fluid(case):
    x = case.store["inputs"][...]
    b = x[4]
    assert b.min() == 0
    # H = sphere diameter; sdf/H at the sphere center = -radius/H = -0.5, so b = 1 + 0.5 / 4 there
    assert 1.11 < b.max() < 1.14
    fluid = b < 1
    assert (x[:4, ~fluid] == 0).all()
    # body voxels are inside the mesh here (immersed boundary), but the STL mask removes them from the fluid
    voxels = case.meta["checks"]["voxels"]
    assert voxels["n_in_mesh_and_body"] > 0 and voxels["proximity"]["n_body_gt1_not_body"] == 0
    np.testing.assert_allclose(x[0][b < 0.75], 1.0, atol=1e-5)    # u / U_inf more than 1 H from the body


def test_ground_proximity_is_a_function_of_the_height_only(case):
    """Ground at z = 0 and H = 0.4: g = 1 - z/H on the layers, 0 from z = H up."""
    gp = case.store["inputs"][5]
    g = case.meta["grid"]
    z = g["origin"][2] + float(g["dx"]) * np.arange(gp.shape[2])
    np.testing.assert_allclose(gp[0, 0], np.maximum(0, 1 - z / case.meta["geometry"]["H"]), atol=1e-6)
    assert (gp == gp[:1, :1, :]).all()


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
