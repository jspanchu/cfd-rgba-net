"""Verification: sliding-window inference and the metrics."""

import numpy as np
import pytest
import torch

from cfd_rgba_net.evaluate import class_counts, metrics, pool, predict, score


def test_windowed_inference_matches_whole_volume():
    """A pointwise net gives the same answer however the volume is cut into windows."""
    torch.manual_seed(0)
    model = torch.nn.Conv3d(3, 3, 1).eval()
    x = np.random.default_rng(0).normal(size=(3, 130, 70, 50)).astype(np.float32)  # needs several padded windows
    whole = model(torch.from_numpy(x)[None]).softmax(1)[0]
    assert (predict(model, torch.from_numpy(x), torch.device("cpu")) - whole).abs().max() < 1e-4


def test_metrics_match_hand_built_masks():
    ref = np.zeros((12, 12, 12), bool)
    ref[2:6, 2:6, 2:6] = True  # 64 voxels
    shifted = np.zeros_like(ref)
    shifted[2:6, 2:6, 3:7] = True  # 48 of them shared
    far = np.zeros_like(ref)
    far[8:12, 8:12, 8:12] = True  # no overlap and no boundary within 1 voxel of ref's
    everywhere = np.ones_like(ref)
    m = metrics(class_counts(shifted, ref, everywhere))
    assert m["iou"] == pytest.approx(48 / 80) and m["dice"] == pytest.approx(0.75)
    assert metrics(class_counts(ref, ref, everywhere))["surface_dice"] == 1
    assert metrics(class_counts(far, ref, everywhere)) == dict(iou=0, dice=0, surface_dice=0)


def test_score_counts_each_region_and_pools_runs():
    shape = (8, 8, 8)
    core, proximity = np.ones(shape, bool), np.zeros(shape)
    proximity[:4] = 1.0  # the first half is near the body, the rest is beyond 4 H
    ref, pred = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    ref[2:6] = 1  # vortex over x = 2..5
    pred[:6] = 1  # the net's vortex also covers x = 0..1
    s = score(pred, ref, core, proximity)
    assert [s[r]["vortex"]["inter"] for r in ("all", "near_body", "far")] == [4 * 64, 2 * 64, 2 * 64]
    assert pool([s, s])["all"]["vortex"]["inter"] == 2 * s["all"]["vortex"]["inter"]
    assert s["all"]["shear_layer"]["ref"] == 0
