"""Verification: the patch loader draws the same patches for a seed and centres them on the classes in thirds."""

from itertools import islice

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cfd_rgba_net.geometry import ImageSpec
from cfd_rgba_net.io import save_case_zarr
from cfd_rgba_net.loader import PatchDataset

N = 40  # voxels per side: smaller than a patch, so every patch is partly zero padding


def write_store(path, rng, n_vortex):
    """A store with a body block, `n_vortex` vortex voxels and 60 shear-layer voxels."""
    inputs = rng.normal(size=(6, N, N, N)).astype(np.float32)
    inputs[4] = 0.0
    inputs[4, 15:20, 15:20, 15:20] = 1.5  # the body
    inputs[5] = np.maximum(0, 1 - np.arange(N) / 32)[None, None, :]  # ground_proximity from the height
    labels = np.zeros((2, N, N, N), np.uint8)
    for channel, count in enumerate((n_vortex, 60)):
        labels[channel].flat[rng.choice(N**3, count, replace=False)] = 1
    save_case_zarr(path, ImageSpec((0, 0, 0), 1 / 32, (N, N, N)), {"inputs": inputs, "labels": labels}, {}, {})


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "t.yaml").write_text("dataset: T\nfiles: {}\nfields: {velocity: [U], pressure: [p]}\n"
                                     "field_source: point\nflow: {U_inf: 1.0, nu: 1.0e-5}\n")
    (tmp_path / "splits.csv").write_text("run,split\n1,train\n2,train\n3,val\n")  # run 3 has no store
    write_store(tmp_path / "t_run1.zarr", rng, n_vortex=30)
    write_store(tmp_path / "t_run2.zarr", rng, n_vortex=0)
    return lambda seed=0: PatchDataset("train", tmp_path / "splits.csv", tmp_path, tmp_path / "index",
                                       dataset=str(tmp_path / "t.yaml"), seed=seed)


def test_same_seed_gives_the_same_first_100_patches(dataset):
    def first_100(seed):
        return list(islice(DataLoader(dataset(seed), batch_size=4, num_workers=2), 25))

    a, b, other = first_100(7), first_100(7), first_100(8)
    assert all(torch.equal(x[k], y[k]) for x, y in zip(a, b) for k in x)
    assert not torch.equal(a[0]["inputs"], other[0]["inputs"])


def test_centres_are_a_third_each_and_a_run_without_vortex_gives_its_share_away(dataset):
    ds, rng = dataset(), np.random.default_rng(0)
    draws = [ds.sample_centre(rng)[:2] for _ in range(3000)]
    share = lambda run, kind: np.mean([k == kind for r, k in draws if r == run])  # noqa: E731
    assert [share(1, k) for k in ("vortex", "shear_layer", "fluid")] == pytest.approx([1 / 3] * 3, abs=0.06)
    assert [share(2, k) for k in ("vortex", "shear_layer", "fluid")] == pytest.approx([0, 1 / 2, 1 / 2], abs=0.06)
