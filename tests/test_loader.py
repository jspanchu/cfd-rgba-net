"""Verification: the patch loader draws the same patches for a seed and centres
them on the classes in thirds."""

from itertools import islice

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from cfd_rgba_net.geometry import ImageSpec
from cfd_rgba_net.io import save_case_zarr
from cfd_rgba_net.loader import PatchDataset

# voxels per side: smaller than a patch, so every patch is partly zero padding
N = 40


def write_store(path, rng, n_vortex):
    """A store with a body block, `n_vortex` vortex voxels and 60 shear-layer
    voxels."""
    body_sdf = np.full((N, N, N), 1.0, np.float32)
    body_sdf[15:20, 15:20, 15:20] = -0.1  # the body
    height = np.broadcast_to(
        np.arange(N, dtype=np.float32) / 32, (N, N, N)
    ).copy()
    labels = np.zeros((2, N, N, N), np.uint8)
    for channel, count in enumerate((n_vortex, 60)):
        labels[channel].flat[rng.choice(N**3, count, replace=False)] = 1
    features = rng.normal(size=(3, N, N, N)).astype(np.float32)
    save_case_zarr(
        path,
        ImageSpec((0, 0, 0), 1 / 32, (N, N, N)),
        {
            "features": features,
            "labels": labels,
            "body_sdf": body_sdf,
            "height": height,
        },
        {},
        {},
    )


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "t.yaml").write_text(
        "dataset: T\nfiles: {}\nfields: {velocity: [U], pressure: [p]}\n"
        "field_source: point\nflow: {U_inf: 1.0, nu: 1.0e-5}\n"
    )
    (tmp_path / "splits.csv").write_text(
        "run,split\n1,train\n2,train\n3,val\n"
    )  # run 3 has no store
    write_store(tmp_path / "t_run1.zarr", rng, n_vortex=30)
    write_store(tmp_path / "t_run2.zarr", rng, n_vortex=0)
    return lambda seed=0: PatchDataset(
        "train",
        tmp_path / "splits.csv",
        tmp_path,
        tmp_path / "index",
        dataset=str(tmp_path / "t.yaml"),
        seed=seed,
    )


def test_same_seed_gives_the_same_first_20_patches(dataset):
    def first_20(seed):
        # fork: the default forkserver re-imports torch in every worker, about 5
        # s per loader
        loader = DataLoader(
            dataset(seed),
            batch_size=4,
            num_workers=2,
            multiprocessing_context="fork",
        )
        return list(islice(loader, 5))

    a, b, other = first_20(7), first_20(7), first_20(8)
    assert all(torch.equal(x[k], y[k]) for x, y in zip(a, b) for k in x)
    assert not torch.equal(a[0]["inputs"], other[0]["inputs"])


def test_unaugmented_patch_is_the_stores_features_around_the_centre(tmp_path):
    rng = np.random.default_rng(1)
    (tmp_path / "t.yaml").write_text(
        "dataset: T\nfiles: {}\nfields: {velocity: [U], pressure: [p]}\n"
        "field_source: point\nflow: {U_inf: 1.0, nu: 1.0e-5}\n"
    )
    (tmp_path / "splits.csv").write_text("run,split\n1,train\n")
    write_store(tmp_path / "t_run1.zarr", rng, n_vortex=30)
    ds = PatchDataset(
        "train",
        tmp_path / "splits.csv",
        tmp_path,
        tmp_path / "index",
        dataset=str(tmp_path / "t.yaml"),
        augment=False,
    )
    _, _, centre = ds.sample_centre(np.random.default_rng(5))
    patch = ds.sample_patch(np.random.default_rng(5))["inputs"].numpy()

    features = zarr.open_group(tmp_path / "t_run1.zarr", mode="r")["features"][
        :
    ]
    # the patch is 96 wide and centred on the voxel; the store is smaller, so it
    # is padded
    lo = centre - 48
    src = tuple(slice(max(l, 0), min(l + 96, N)) for l in lo)
    dst = tuple(slice(s.start - l, s.stop - l) for s, l in zip(src, lo))
    assert patch.shape == (3, 96, 96, 96)
    np.testing.assert_array_equal(
        patch[(slice(None),) + dst], features[(slice(None),) + src]
    )


def test_centres_are_a_third_each_and_a_run_without_vortex_gives_its_share_away(
    dataset,
):
    ds, rng = dataset(), np.random.default_rng(0)
    draws = [ds.sample_centre(rng)[:2] for _ in range(3000)]
    share = lambda run, kind: np.mean([k == kind for r, k in draws if r == run])
    assert [
        share(1, k) for k in ("vortex", "shear_layer", "fluid")
    ] == pytest.approx([1 / 3] * 3, abs=0.06)
    assert [
        share(2, k) for k in ("vortex", "shear_layer", "fluid")
    ] == pytest.approx([0, 1 / 2, 1 / 2], abs=0.06)
