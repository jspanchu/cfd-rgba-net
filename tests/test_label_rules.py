"""Verification: hand-built voxel fixtures for each label rule (exact expected masks)."""

import numpy as np
import pytest

from cfd_rgba_net.config import Axis, LabelRules
from cfd_rgba_net.geometry import CaseGeometry, ImageSpec
from cfd_rgba_net.label import CLASS_NAMES, make_labels

N_PER_H = 32
H = 1.0
DX = H / N_PER_H
DIMS = (40, 20, 20)
SPEC = ImageSpec(origin=(0.0, 0.0, 0.0), dx=DX, dims=DIMS)
# ground at z = 0; the body bounds only feed CaseGeometry
GEO = CaseGeometry(body_bounds=(-1.0, 9.5 * DX, 0.0, 1.0, 0.0, 1.0), ground=0.0, H=H, up=Axis(2, 1))
RULES = LabelRules.load()
FLOOR_K = int(np.ceil(RULES.floor_exclude_H * N_PER_H))  # first voxel layer above the floor exclusion
Z_OK = FLOOR_K + 3                                       # a layer safely inside the core
IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def quiet():
    """Ingredients for which no rule fires."""
    z = np.zeros(DIMS, np.float32)
    return dict(Q_nd=z.copy(), lambda2_nd=z.copy(), omega_mag_nd=z.copy())


def fluid_all():
    return np.ones(DIMS, bool)


def labels(ing, fluid=None):
    return make_labels(ing, fluid_all() if fluid is None else fluid, SPEC, GEO, RULES, N_PER_H)


def test_quiet_flow_has_no_labels():
    r = labels(quiet())
    assert r.labels.sum() == 0
    assert r.fractions["none"] == 1.0


def test_vortex_needs_lambda2_and_q():
    ing = quiet()
    p = (20, 10, Z_OK)
    ing["lambda2_nd"][p] = -RULES.tau_lambda2 - 0.1
    ing["Q_nd"][p] = 0.1
    q = (22, 10, Z_OK)                       # lambda2 fires but Q <= 0
    ing["lambda2_nd"][q] = -RULES.tau_lambda2 - 0.1
    s = (24, 10, Z_OK)                       # at the threshold: strict inequality, no label
    ing["lambda2_nd"][s] = -RULES.tau_lambda2
    ing["Q_nd"][s] = 0.1
    v = labels(ing).labels[IDX["vortex"]]
    assert v[p] == 1 and v[q] == 0 and v[s] == 0
    assert v.sum() == 1


def test_shear_excludes_vortex():
    ing = quiet()
    a, b = (20, 10, Z_OK), (22, 10, Z_OK)
    ing["omega_mag_nd"][a] = ing["omega_mag_nd"][b] = RULES.tau_omega + 1
    ing["lambda2_nd"][b] = -RULES.tau_lambda2 - 1
    ing["Q_nd"][b] = 1.0
    r = labels(ing).labels
    assert r[IDX["shear_layer"]][a] == 1 and r[IDX["vortex"]][a] == 0
    assert r[IDX["shear_layer"]][b] == 0 and r[IDX["vortex"]][b] == 1
    assert not (r[IDX["shear_layer"]] & r[IDX["vortex"]]).any()


def test_floor_exclusion():
    """Nothing below floor_exclude_H * H is labeled, and everything above it is, away from the grid's faces."""
    ing = quiet()
    ing["omega_mag_nd"][:] = RULES.tau_omega + 1
    s = labels(ing).labels[IDX["shear_layer"]]
    z = SPEC.axis(2)
    below = z < RULES.floor_exclude_H * H
    assert s[:, :, below].sum() == 0
    # stay clear of the grid's faces, where the wall erosion removes the outer layers (the top one, in z)
    assert s[5:-5, 5:-5, ~below][:, :, :-2].all()


def test_wall_erosion():
    """Labels skip the voxel layer next to any non-fluid voxel."""
    fluid = fluid_all()
    fluid[15:25, 5:15, 10:20] = False         # a solid block
    ing = quiet()
    ing["omega_mag_nd"][:] = RULES.tau_omega + 1
    r = labels(ing, fluid)
    s = r.labels[IDX["shear_layer"]]
    assert s[14, 10, 15] == 0 and s[12, 10, 15] == 1   # adjacent to the block vs 3 voxels away
    assert (r.labels[:, ~fluid] == 0).all()


def test_labels_skip_the_grids_outer_layer():
    """The erosion treats the outside of the grid as non-fluid, so the grid's faces act as walls."""
    ing = quiet()
    ing["omega_mag_nd"][:] = RULES.tau_omega + 1
    s = labels(ing).labels[IDX["shear_layer"]]
    for face in (s[0, :, Z_OK], s[-1, :, Z_OK], s[:, 0, Z_OK], s[:, -1, Z_OK]):
        assert not face.any()
    assert s[1, 10, Z_OK] == 1 and s[-2, 10, Z_OK] == 1


def test_class_fractions_are_over_all_fluid_voxels():
    """Fluid voxels excluded from labeling (floor, wall layer) count as fluid with no label."""
    ing = quiet()
    vortex_voxels = [(20, 10, Z_OK), (22, 10, Z_OK)]
    shear_voxels = [(24, 10, Z_OK), (26, 10, Z_OK), (28, 10, Z_OK)]
    for p in vortex_voxels:
        ing["lambda2_nd"][p] = -RULES.tau_lambda2 - 0.1
        ing["Q_nd"][p] = 0.1
    for p in shear_voxels:
        ing["omega_mag_nd"][p] = RULES.tau_omega + 1
    n = int(np.prod(DIMS))
    assert labels(ing).fractions == pytest.approx(
        {"vortex": 2 / n, "shear_layer": 3 / n, "none": (n - 5) / n})
