"""Verification: the baseline thresholds the net's grid features, which do not depend on the grid's orientation."""

import numpy as np
import torch

from cfd_rgba_net.baseline import baseline_classes, grid_features, lambda2_closed_form
from cfd_rgba_net.config import LabelRules
from cfd_rgba_net.ingredients import lambda2

H = 1 / 32


def _synthetic_field(n=32):
    """A smooth random velocity field around a block-shaped body, and the fluid mask."""
    g = np.random.default_rng(0)
    u = g.normal(size=(3, n + 4, n + 4, n + 4))
    for axis in (1, 2, 3):  # smooth with a 5-voxel box filter along each axis
        u = sum(np.roll(u, k, axis) for k in range(-2, 3)) / 5
    u = (u[:, 2:-2, 2:-2, 2:-2] * 2).astype(np.float32)
    fluid = np.ones((n,) * 3, bool)
    fluid[10:18, 10:18, 10:18] = False  # body
    u[:, ~fluid] = 0
    return u, fluid


def test_baseline_thresholds_the_features_by_the_rules_inside_the_core():
    u, fluid = _synthetic_field()
    features, rules = grid_features(u, fluid, H), LabelRules.load()
    core = fluid.copy()
    core[:3] = False  # the core is a subset of the grid

    classes = baseline_classes(features, core, rules)
    lam2, Q, omega = features
    vortex = (lam2 < -rules.tau_lambda2) & (Q > 0)
    expected = np.where(core, vortex + 2 * ((omega > rules.tau_omega) & ~vortex), 0)
    assert (classes == 1).any() and (classes == 2).any()  # the field exercises both classes
    assert (classes[~core] == 0).all()
    np.testing.assert_array_equal(classes, expected)


def test_features_are_invariant_under_orientation():
    u, fluid = _synthetic_field()
    # rotate 90° about z: the grid axes (x, y) -> (-y, x), and so do the velocity components
    rot = np.rot90(u, 1, axes=(1, 2)).copy()
    rot[[0, 1]] = np.stack([-rot[1], rot[0]])
    np.testing.assert_allclose(grid_features(rot, np.rot90(fluid, 1, axes=(0, 1)), H),
                               np.rot90(grid_features(u, fluid, H), 1, axes=(1, 2)), atol=1e-4)


def test_closed_form_lambda2_matches_the_labels_eigenvalue_solver():
    G = np.random.default_rng(0).normal(size=(10_000, 3, 3)).astype(np.float32) * 5
    G[0] = 0  # three equal eigenvalues
    G[1] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]  # pure rotation: a repeated eigenvalue
    G[2] = [[0, 1, 0], [0, 0, 0], [0, 0, 0]]  # pure shear: lambda2 = 0
    np.testing.assert_allclose(lambda2_closed_form(torch.from_numpy(G)).numpy(), lambda2(G), rtol=1e-5, atol=1e-5)
