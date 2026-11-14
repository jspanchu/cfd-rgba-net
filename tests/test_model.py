"""Verification: the UNet keeps the grid size and returns 3 class scores, plus deep-supervision outputs in training."""

import pytest
import torch

from cfd_rgba_net.model import UNet3D


@pytest.mark.parametrize("levels, norm", [(5, "instance"), (4, "instance"), (5, "group"), (5, "batch")])
def test_output_shapes_in_eval_and_training(levels, norm):
    model = UNet3D(levels=levels, norm=norm)
    x = torch.randn(2, 6, 32, 48, 16)  # sides are multiples of 16 and need not be equal

    model.eval()
    assert model(x).shape == (2, 3, 32, 48, 16)

    model.train()
    full, half, quarter = model(x)
    assert [o.shape[2:] for o in (full, half, quarter)] == [(32, 48, 16), (16, 24, 8), (8, 12, 4)]
