"""Verification: the body and ground proximity channels (closed form, continuity, inside vs outside)."""

import numpy as np
import pytest

from cfd_rgba_net.config import Contract
from cfd_rgba_net.preprocess import INPUT_CHANNELS, body_proximity, ground_proximity


@pytest.mark.parametrize("sdf_H,expected", [
    (-0.5, 1.125),   # inside the body the line continues above 1
    (0.0, 1.0),      # on the surface
    (1.0, 0.75),
    (2.0, 0.5),
    (4.0, 0.0),      # 4 H out
    (10.0, 0.0),     # beyond: clamped to 0
])
def test_body_proximity_values(sdf_H, expected):
    assert body_proximity(np.array([sdf_H]))[0] == pytest.approx(expected, abs=1e-7)


def test_body_proximity_continuous_and_signed():
    """One line from inside to 4 H out: no jump at the surface, b > 1 exactly where sdf < 0."""
    s = np.linspace(-1.0, 6.0, 7001, dtype=np.float32)
    b = body_proximity(s)
    assert b.dtype == np.float32
    assert np.abs(np.diff(b)).max() <= 1.001 * (s[1] - s[0]) / 4      # slope 1/4 everywhere, 0 past 4 H
    assert ((b > 1) == (s < 0)).all() and ((b < 1) == (s > 0)).all()
    assert (b[s >= 4] == 0).all() and (b[s < 4] > 0).all()


@pytest.mark.parametrize("h_H,expected", [(0.0, 1.0), (1e-6, 1 - 1e-6), (0.25, 0.75), (1.0, 0.0), (3.0, 0.0)])
def test_ground_proximity_values(h_H, expected):
    assert ground_proximity(np.array([h_H]))[0] == pytest.approx(expected, abs=1e-7)


def test_contract_channels_match_preprocess():
    assert Contract.load().channels == INPUT_CHANNELS
