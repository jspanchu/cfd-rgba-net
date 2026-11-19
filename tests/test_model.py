"""Verification: the UNet keeps the grid size for any number of input
channels and classes."""

import pytest
import torch

from cfd_rgba_net.model import UNet3D


@pytest.mark.parametrize(
    "levels, n_in, n_out", [(2, 3, 3), (4, 1, 5), (5, 3, 3)]
)
def test_output_shapes_in_eval_and_training(levels, n_in, n_out):
    model = UNet3D(n_in, n_out, levels)
    # sides are multiples of 16 and need not be equal
    x = torch.rand(2, n_in, 32, 48, 16) * 2 - 1
    for mode in (model.eval, model.train):
        mode()
        assert model(x).shape == (2, n_out, 32, 48, 16)
