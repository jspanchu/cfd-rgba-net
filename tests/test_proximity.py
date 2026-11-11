"""Verification: the body and ground proximity channels (closed form, continuity, inside vs outside)."""

import numpy as np

from cfd_rgba_net.config import Contract
from cfd_rgba_net.preprocess import INPUT_CHANNELS, body_proximity, ground_proximity


def test_body_proximity():
    """One line from inside to 4 H out: no jump at the surface, b > 1 exactly where sdf < 0."""
    # inside the line continues above 1; 1 on the surface; 0 from 4 H out
    np.testing.assert_allclose(body_proximity(np.array([-0.5, 0.0, 4.0, 10.0])), [1.125, 1.0, 0.0, 0.0], atol=1e-7)
    s = np.linspace(-1.0, 6.0, 7001, dtype=np.float32)
    b = body_proximity(s)
    assert b.dtype == np.float32
    assert np.abs(np.diff(b)).max() <= 1.001 * (s[1] - s[0]) / 4      # slope 1/4 everywhere, 0 past 4 H
    assert ((b > 1) == (s < 0)).all() and ((b < 1) == (s > 0)).all()
    assert (b[s >= 4] == 0).all() and (b[s < 4] > 0).all()


def test_ground_proximity():
    np.testing.assert_allclose(ground_proximity(np.array([0.0, 1e-6, 0.25, 1.0, 3.0])),
                               [1.0, 1 - 1e-6, 0.75, 0.0, 0.0], atol=1e-7)


def test_contract_channels_match_preprocess():
    assert Contract.load().channels == INPUT_CHANNELS
