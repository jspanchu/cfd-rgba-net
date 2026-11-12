"""Verification: the loss ignores voxels outside the core and stays finite when a batch has no labelled voxel."""

import pytest
import torch

from cfd_rgba_net.loss import DeepSupervisionLoss, class_weights

SHAPES = (8, 4, 2)  # full, half and quarter resolution sides


def make_batch(labelled=True):
    labels = torch.zeros(2, 2, 8, 8, 8, dtype=torch.uint8)
    if labelled:
        labels[:, 0, :4] = 1  # vortex
        labels[:, 1, 4:6] = 1  # shear layer
    core = torch.rand(2, 8, 8, 8) > 0.3
    return labels, core


def perfect_logits(labels):
    target = (labels[:, 0] + 2 * labels[:, 1]).long()
    return [torch.nn.functional.one_hot(target[:, ::2**i, ::2**i, ::2**i], 3).movedim(-1, 1) * 50.0
            for i in range(3)]


@pytest.fixture
def loss_fn():
    return DeepSupervisionLoss(class_weights((0.017, 0.029)))


def test_class_weights_are_capped_inverse_frequencies():
    assert class_weights((0.017, 0.029)) == pytest.approx(torch.tensor([1.0, 57.8, 33.5]), abs=0.1)
    assert class_weights((0.001, 0.5)) == pytest.approx(torch.tensor([1.0, 100.0, 1.0]))


def test_perfect_prediction_gives_zero_loss(loss_fn):
    labels, core = make_batch()
    assert loss_fn(perfect_logits(labels), labels, core).item() == pytest.approx(0, abs=1e-4)


def test_values_outside_the_core_do_not_change_the_loss(loss_fn):
    labels, core = make_batch()
    logits = [torch.randn(2, 3, *(s,) * 3) for s in SHAPES]
    before = loss_fn(logits, labels, core)
    for i, x in enumerate(logits):
        outside = ~core[:, ::2**i, ::2**i, ::2**i]
        x.permute(0, 2, 3, 4, 1)[outside] = 100.0
    labels[:, 0][~core] = 1  # so are the labels (vortex here, never both classes at once)
    labels[:, 1][~core] = 0
    assert loss_fn(logits, labels, core).item() == pytest.approx(before.item())


def test_gradients_are_finite_without_labelled_voxels(loss_fn):
    labels, core = make_batch(labelled=False)
    logits = [torch.randn(2, 3, *(s,) * 3, requires_grad=True) for s in SHAPES]
    loss_fn(logits, labels, core).backward()
    assert all(torch.isfinite(x.grad).all() for x in logits)
