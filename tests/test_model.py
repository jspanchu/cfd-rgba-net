"""Verification: the UNet keeps the grid size, and with the prior its untrained
form reproduces the rules."""

import numpy as np
import pytest
import torch

from cfd_rgba_net.baseline import grid_features
from cfd_rgba_net.model import UNet3D

TAU_LAMBDA2, TAU_OMEGA = 0.5, 2.0


@pytest.mark.parametrize("levels", [2, 4, 5])
def test_output_shapes_in_eval_and_training(levels):
    model = UNet3D(TAU_LAMBDA2, TAU_OMEGA, levels)
    x = torch.randn(
        2, 3, 32, 48, 16
    )  # sides are multiples of 16 and need not be equal
    for mode in (model.eval, model.train):
        mode()
        assert model(x).shape == (2, 3, 32, 48, 16)


def _synthetic_features(n=32):
    """Features of a smooth random velocity field around a block-shaped body."""
    g = np.random.default_rng(0)
    u = (
        torch.nn.functional.avg_pool3d(
            torch.from_numpy(g.normal(size=(1, 3, n, n, n)).astype(np.float32)),
            5,
            1,
            2,
        )[0].numpy()
        * 2
    )
    fluid = np.ones((n,) * 3, bool)
    fluid[10:18, 10:18, 10:18] = False
    u[:, ~fluid] = 0
    return torch.from_numpy(grid_features(u, fluid, 1 / 32))[None]


def test_untrained_prior_model_reproduces_the_rules():
    f = _synthetic_features()
    pred = (
        UNet3D(TAU_LAMBDA2, TAU_OMEGA, prior=True)
        .eval()(f)
        .argmax(1)[0]
        .numpy()
    )

    lam2, Q, omega = f[0].numpy()
    vortex = (lam2 < -TAU_LAMBDA2) & (Q > 0)
    expected = vortex + 2 * ((omega > TAU_OMEGA) & ~vortex)
    assert (
        expected.any() and (expected == 2).any()
    )  # the field exercises both classes
    assert (
        pred == expected
    ).mean() > 0.999  # float round-off only flips voxels sitting on a threshold


def test_prior_model_trains_through_the_zero_initialised_head():
    model = UNet3D(TAU_LAMBDA2, TAU_OMEGA, prior=True).train()
    for _ in range(
        2
    ):  # the first step moves only the head, the second reaches the encoder
        model.zero_grad()
        model(_synthetic_features()).square().mean().backward()
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 0.1 * p.grad
    assert model.enc[0][0].weight.grad.abs().sum() > 0
