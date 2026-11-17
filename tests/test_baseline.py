"""Verification: the net's grid features carry the baseline's rules and do not depend on the grid's orientation."""

import numpy as np

from cfd_rgba_net.baseline import baseline_classes, grid_features
from cfd_rgba_net.config import Contract, LabelRules

H = 1 / 32


def _synthetic_inputs(n=32):
    """Unscaled store inputs: a smooth random velocity field around a block-shaped body."""
    g = np.random.default_rng(0)
    x = np.zeros((6, n, n, n), np.float32)
    u = g.normal(size=(3, n + 4, n + 4, n + 4))
    for axis in (1, 2, 3):  # smooth with a 5-voxel box filter along each axis
        u = sum(np.roll(u, k, axis) for k in range(-2, 3)) / 5
    x[:3] = u[:, 2:-2, 2:-2, 2:-2] * 2
    x[4] = 0.5  # fluid
    x[4, 10:18, 10:18, 10:18] = 2  # body
    x[:3, 10:18, 10:18, 10:18] = 0
    return x


def test_features_thresholded_by_the_rules_give_the_baseline():
    x, contract, rules = _synthetic_inputs(), Contract.load(), LabelRules.load()
    lam2, Q, omega = grid_features(x[:3], x[4] < 1, 1 / contract.n_per_H)
    vortex = (lam2 < -rules.tau_lambda2) & (Q > 0)
    from_features = vortex + 2 * ((omega > rules.tau_omega) & ~vortex)

    expected = baseline_classes(x, np.ones(x.shape[1:], bool), contract, rules)
    assert (expected == 1).any() and (expected == 2).any()  # the field exercises both classes
    np.testing.assert_array_equal(from_features, expected)


def test_features_are_invariant_under_orientation():
    x = _synthetic_inputs()
    # rotate 90° about z: the grid axes (x, y) -> (-y, x), and so do the velocity components
    rot = np.rot90(x, 1, axes=(1, 2)).copy()
    rot[[0, 1]] = np.stack([-rot[1], rot[0]])
    np.testing.assert_allclose(grid_features(rot[:3], rot[4] < 1, H),
                               np.rot90(grid_features(x[:3], x[4] < 1, H), 1, axes=(1, 2)), atol=1e-4)
