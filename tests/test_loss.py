"""Verification: the loss ignores voxels outside the core and stays finite when a batch has no labelled voxel."""

import pytest
import torch

from cfd_rgba_net.loss import SegmentationLoss, class_weights


def make_batch(labelled=True):
    labels = torch.zeros(2, 2, 8, 8, 8, dtype=torch.uint8)
    if labelled:
        labels[:, 0, :4] = 1  # vortex
        labels[:, 1, 4:6] = 1  # shear layer
    core = torch.rand(2, 8, 8, 8) > 0.3
    return labels, core


@pytest.fixture
def loss_fn():
    return SegmentationLoss(class_weights((0.017, 0.029)))


def test_class_weights_are_capped_inverse_frequencies():
    assert class_weights((0.017, 0.029)) == pytest.approx(torch.tensor([1.0, 57.8, 33.5]), abs=0.1)
    assert class_weights((0.001, 0.5)) == pytest.approx(torch.tensor([1.0, 100.0, 1.0]))


def test_values_outside_the_core_do_not_change_the_loss(loss_fn):
    labels, core = make_batch()
    logits = torch.randn(2, 3, 8, 8, 8)
    before = loss_fn(logits, labels, core)[0]
    logits.permute(0, 2, 3, 4, 1)[~core] = 100.0
    labels[:, 0][~core] = 1  # so are the labels (vortex here, never both classes at once)
    labels[:, 1][~core] = 0
    assert loss_fn(logits, labels, core)[0].item() == pytest.approx(before.item())


def test_gradients_are_finite_without_labelled_voxels(loss_fn):
    labels, core = make_batch(labelled=False)
    logits = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
    loss_fn(logits, labels, core)[0].backward()
    assert torch.isfinite(logits.grad).all()
