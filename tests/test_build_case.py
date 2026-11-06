"""
End to end on a tiny synthetic case: a sphere inside a hexahedral mesh that also fills the sphere
(immersed-boundary layout), uniform flow plus a Lamb-Oseen vortex along x through y = 0, z = 0.6.
"""

import numpy as np
import pytest
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOGeometry import vtkSTLWriter
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

from cfd_rgba_net.build_case import build_case
import zarr

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
flow:
  U_inf: 10.0
  p_inf: 0.0
  rho: 1.0
  nu: 1.0e-5
  up_direction: [0, 0, 1]
  ground: auto
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
    vtkXMLUnstructuredGridWriter(file_name=str(d / "volume_1.vtu"), input_data=ug).Write()
    (root / "synthetic.yaml").write_text(DATASET_YAML)


def test_build_case_fails_without_body_voxels(tmp_path):
    """A domain that misses the body is an error, and no store is written."""
    make_case(tmp_path)
    yaml = (tmp_path / "synthetic.yaml")
    yaml.write_text(DATASET_YAML + "domain: [1.5, 3.0, -0.75, 0.75, 0.0, 1.5]\n")
    cfg = CaseConfig.load(yaml)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no voxel lies inside the body"):
        build_case(cfg, 1, tmp_path, out, Contract("1.0", 16, ("u", "v", "w", "Cp", "sdf")), LabelRules.load(),
                   log=lambda m: None)
    assert not (out / "synthetic_run1.zarr").exists()


def test_build_case(tmp_path):
    make_case(tmp_path)
    cfg = CaseConfig.load(tmp_path / "synthetic.yaml")
    contract = Contract("1.0", 16, ("u", "v", "w", "Cp", "sdf"))
    out = tmp_path / "out"
    meta = build_case(cfg, 1, tmp_path, out, contract, LabelRules.load(), save_vti=True, log=lambda m: None)

    store = zarr.open_group(out / "synthetic_run1.zarr", mode="r")
    m = dict(store.attrs)
    assert not (out / "synthetic_run1.zarr.tmp").exists()
    x = store["inputs"][...]
    assert x.shape[0] == 5 and m["channel_names"] == ["u", "v", "w", "Cp", "sdf"]
    assert store["inputs"].chunks == (5, 64, 64, 64) and x.dtype == np.float32
    fluid = store["fluid"][...]
    # the grid fills the mesh domain (x -1..3, y -0.75..0.75, z 0..1.5), just inside its faces
    g = meta["grid"]
    assert -1.0 < g["origin"][0] < -1.0 + 1e-3 and 0.0 < g["origin"][2] < 1e-3
    sdf = x[4]
    # body voxels are inside the mesh here, but the STL mask removes them from the fluid
    assert meta["checks"]["voxels"]["n_in_mesh_and_body"] > 0
    assert not fluid[sdf < 0].any()
    assert (x[:4, ~fluid] == 0).all()
    np.testing.assert_allclose(x[0][fluid & (sdf > 1.0)], 1.0, atol=1e-5)  # u/U_inf far from the body
    # H = sphere diameter; sdf/H at the sphere center = -radius/H = -0.5
    assert sdf.min() == np.float32(sdf.min()) and -0.55 < sdf.min() < -0.45

    L = store["labels"][...]
    names = m["class_names"]
    vortex = L[names.index("vortex")]
    # vortex voxels hug the vortex axis, and the axis behind the body is labelled
    o, dx = m["grid"]["origin"], float(m["grid"]["dx"])
    i, j, k = np.nonzero(vortex)
    r = np.hypot(o[1] + dx * j, o[2] + dx * k - 0.6)
    assert vortex.sum() > 0 and (r < 2 * RC).all()
    xs = o[0] + dx * np.arange(vortex.shape[0])
    jy, kz = round(-o[1] / dx), round((0.6 - o[2]) / dx)
    axis = vortex[(xs > 0.5) & (xs < 2.5), jy, kz]
    assert axis.mean() > 0.9
    assert names == ["vortex", "shear_layer"]

    assert set(store["ingredients"].array_keys()) == {"Q_nd", "lambda2_nd", "omega_mag_nd", "core"}
    assert m["provenance"]["contract_version"] == "1.0"
    assert m["provenance"]["label_rules_version"] == "1"
    assert len(m["provenance"]["inputs"]["volume"]["sha256"]) == 64
    assert m["geometry"]["H"] == np.float64(m["geometry"]["H"]) and abs(m["geometry"]["H"] - 0.4) < 1e-3
    for f in ("inputs", "ingredients", "labels"):
        assert (out / f"synthetic_run1_{f}.vti").exists()

