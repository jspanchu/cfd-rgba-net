"""Verification: the UNet keeps the grid size and returns 3 class scores, plus deep-supervision outputs in training."""

import numpy as np
import pytest
import torch

from cfd_rgba_net.baseline import velocity_gradient
from cfd_rgba_net.ingredients import lambda2, q_and_vorticity
from cfd_rgba_net.model import DerivedFeatures, PlainUNet3D, UNet3D


@pytest.mark.parametrize("levels, norm", [(5, "instance"), (4, "instance"), (5, "group"), (5, "batch")])
def test_output_shapes_in_eval_and_training(levels, norm):
    model = UNet3D(levels=levels, norm=norm)
    x = torch.randn(2, 6, 32, 48, 16)  # sides are multiples of 16 and need not be equal

    model.eval()
    assert model(x).shape == (2, 3, 32, 48, 16)

    model.train()
    full, half, quarter = model(x)
    assert [o.shape[2:] for o in (full, half, quarter)] == [(32, 48, 16), (16, 24, 8), (8, 12, 4)]


def _synthetic_inputs(n=32):
    """Scaled inputs with a smooth random velocity field and a block-shaped body."""
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(1, 6, n, n, n)
    smooth = torch.nn.functional.avg_pool3d(torch.randn(1, 3, n, n, n, generator=g), 5, 1, 2)
    x[:, :3] = smooth * 4
    x[:, 4] = 0.5 / DERIVED.body_scale  # fluid
    x[:, 4, 10:18, 10:18, 10:18] = 2 / DERIVED.body_scale  # body
    x[:, :3, 10:18, 10:18, 10:18] = 0
    return x


DERIVED = DerivedFeatures(0.585942, 0.202054, 32, 0.5, 2.0)


def test_derived_features_reproduce_the_baseline():
    x = _synthetic_inputs()
    feats = DERIVED(x)[0, 6:].numpy()

    unscaled = x[0].numpy() * np.array([0.585942] * 3 + [1, 0.202054, 1], np.float32)[:, None, None, None]
    core = np.ones(x.shape[2:], bool)
    G = velocity_gradient(unscaled[:3], unscaled[4] < 1, core, 1 / 32)
    Q, omega = q_and_vorticity(G)
    expected = np.stack([np.tanh((-lambda2(G) - 0.5) / 0.5), np.tanh(Q / 0.5), np.tanh((omega - 2.0) / 2.0)])
    np.testing.assert_allclose(feats.reshape(3, -1), expected.reshape(3, -1), atol=1e-4)


def test_derived_features_are_invariant_under_orientation():
    x = _synthetic_inputs()
    # rotate 90° about z: the grid axes (x, y) -> (-y, x), and so do the velocity components
    rot = torch.rot90(x, 1, dims=(2, 3))
    rot[:, [0, 1]] = torch.stack([-rot[:, 1], rot[:, 0]], dim=1)
    np.testing.assert_allclose(DERIVED(rot)[:, 6:].numpy(), torch.rot90(DERIVED(x)[:, 6:], 1, dims=(2, 3)).numpy(),
                               atol=1e-4)


def test_unet_with_derived_features_keeps_the_output_shape():
    model = UNet3D(features=DERIVED).eval()
    assert model(torch.randn(1, 6, 32, 32, 16)).shape == (1, 3, 32, 32, 16)


def test_untrained_prior_model_reproduces_the_rules():
    x = _synthetic_inputs()
    model = UNet3D(features=DERIVED, prior=True).eval()
    pred = model(x).argmax(1)[0].numpy()

    unscaled = x[0].numpy() * np.array([0.585942] * 3 + [1, 0.202054, 1], np.float32)[:, None, None, None]
    G = velocity_gradient(unscaled[:3], unscaled[4] < 1, np.ones(x.shape[2:], bool), 1 / 32)
    Q, omega = q_and_vorticity(G)
    vortex = (lambda2(G) < -0.5) & (Q > 0)
    expected = (vortex + 2 * ((omega > 2.0) & ~vortex)).reshape(pred.shape)
    assert expected.any() and (expected == 2).any()  # the field exercises both classes
    assert (pred == expected).mean() > 0.999  # float round-off only flips voxels sitting on a threshold


def test_prior_model_trains_through_the_zero_initialised_heads():
    model = UNet3D(features=DERIVED, prior=True).train()
    out = model(_synthetic_inputs())
    assert [o.shape[2:] for o in out] == [(32,) * 3, (16,) * 3, (8,) * 3]
    for _ in range(2):  # the first step moves only the heads, the second reaches the encoder
        model.zero_grad()
        sum(o.square().mean() for o in model(_synthetic_inputs())).backward()
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 0.1 * p.grad
    assert model.stem[0].weight.grad.abs().sum() > 0


def test_prior_needs_features():
    with pytest.raises(ValueError):
        UNet3D(prior=True)


FEATURES_ONLY = DerivedFeatures(0.585942, 0.202054, 32, 0.5, 2.0, keep_inputs=False)


def test_features_only_stem_returns_the_appended_features_alone():
    x = _synthetic_inputs()
    np.testing.assert_array_equal(FEATURES_ONLY(x).numpy(), DERIVED(x)[:, 6:].numpy())


@pytest.mark.parametrize("levels", [2, 4, 5])
def test_plain_unet_output_shapes_in_eval_and_training(levels):
    model = PlainUNet3D(levels=levels)
    x = torch.randn(2, 6, 32, 48, 16)

    model.eval()
    assert model(x).shape == (2, 3, 32, 48, 16)

    model.train()
    (full,) = model(x)  # one head, no deep supervision
    assert full.shape == (2, 3, 32, 48, 16)


@pytest.mark.parametrize("net", [UNet3D, PlainUNet3D])
def test_untrained_features_only_prior_model_matches_the_full_input_one(net):
    # the prior reads the features wherever the stem puts them, so both stems start from the same rules
    x = _synthetic_inputs()
    full = net(features=DERIVED, prior=True).eval()
    only = net(features=FEATURES_ONLY, prior=True).eval()
    expected = full(x).argmax(1)
    assert (expected > 0).any()
    assert torch.equal(only(x).argmax(1), expected)
