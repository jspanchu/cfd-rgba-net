"""
Patch loader (roadmap, Phase 1.1 task 2): batches of 96^3 training patches from the Zarr stores of one split.

    ds = PatchDataset("train", "configs/splits/ahmedml_v1.csv", "/data/.../out_v1", "/data/.../index", seed=0)
    for batch in DataLoader(ds, batch_size=4, num_workers=4):   # an endless stream of batches
        batch["inputs"], batch["labels"], batch["core"]

Each patch is centred on a vortex voxel, a shear-layer voxel or a random fluid voxel, a third of the
time each, then given one of the 48 axis-aligned orientations. The inputs are divided by the fixed
scales ``input_scales`` of configs/contract_v1.yaml, which scripts/ahmedml/compute_scales.py computes.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import IterableDataset, get_worker_info

from .batch import store_path
from .config import CaseConfig, Contract, LabelRules, _load, configs_dir
from .label import core_mask

PATCH = 96       # voxels per patch side: 3 H at H/32
N_FLUID = 20_000  # random fluid voxels in a run's sampling index, about as many as its vortex voxels
KINDS = ("vortex", "shear_layer", "fluid")
# an orientation permutes the axes, then flips some: 6 permutations x 8 flip patterns
ORIENTATIONS = [(p, np.array(f)) for p in itertools.permutations(range(3)) for f in itertools.product((1, -1), repeat=3)]


def build_index(store: zarr.Group, run: int) -> dict[str, np.ndarray]:
    """List the voxels that patches are centred on.

    Parameters
    ----------
    store : zarr.Group
        The run's case store.
    run : int
        The run number. Seeds the choice of the random fluid voxels, so a rebuild gives the same index.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``(n, 3)`` int32 voxel coordinates under each name in ``KINDS``: every vortex voxel, every
        shear-layer voxel, and ``N_FLUID`` fluid voxels (``body_proximity < 1``) chosen at random.
    """
    labels = store["labels"]
    fluid = np.flatnonzero(store["inputs"][4] < 1)
    pick = np.random.default_rng(run).choice(fluid, N_FLUID, replace=False)
    coords = [np.argwhere(labels[0]), np.argwhere(labels[1]), np.stack(np.unravel_index(pick, labels.shape[1:]), 1)]
    return {k: c.astype(np.int32) for k, c in zip(KINDS, coords)}


def load_index(store_dir: Path, run: int, cache: Path) -> dict[str, np.ndarray]:
    """Read a run's sampling index from the cache, building it first if the cache is missing or stale.

    Parameters
    ----------
    store_dir : pathlib.Path
        The run's ``.zarr`` store.
    run : int
        The run number.
    cache : pathlib.Path
        The cache folder, created if missing. The index is ``<cache>/<store name>.npz``.

    Returns
    -------
    dict[str, numpy.ndarray]
        The index, as ``build_index`` returns it.
    """
    path = cache / f"{store_dir.stem}.npz"
    stamp = (store_dir / "zarr.json").stat().st_mtime_ns  # a rebuilt store changes it
    if path.exists():
        with np.load(path) as f:
            if f["stamp"] == stamp:
                return {k: f[k] for k in KINDS}
    index = build_index(zarr.open_group(store_dir, mode="r"), run)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".tmp.npz"), stamp=stamp, **index)
    path.with_suffix(".tmp.npz").rename(path)
    return index


def read_crop(a: zarr.Array, lo: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a cubic crop of the last three axes, padding with zeros where it leaves the grid.

    Parameters
    ----------
    a : zarr.Array
        Array shaped ``(..., nx, ny, nz)``.
    lo : numpy.ndarray
        The crop's lowest voxel on each of the three axes, may be negative.
    size : int
        The crop's side in voxels.

    Returns
    -------
    crop : numpy.ndarray
        Shaped ``(..., size, size, size)``, same dtype as ``a``.
    inside : numpy.ndarray
        Boolean ``(size, size, size)``: True where the crop lies in the grid.
    """
    src = tuple(slice(max(l, 0), min(l + size, n)) for l, n in zip(lo, a.shape[-3:]))
    dst = tuple(slice(s.start - l, s.stop - l) for s, l in zip(src, lo))
    crop = np.zeros(a.shape[:-3] + (size,) * 3, a.dtype)
    crop[(...,) + dst] = a[(...,) + src]
    inside = np.zeros((size,) * 3, bool)
    inside[dst] = True
    return crop, inside


def orient(a: np.ndarray, perm: tuple[int, ...], sign: np.ndarray) -> np.ndarray:
    """Permute, then flip, the last three axes of an array.

    Parameters
    ----------
    a : numpy.ndarray
        Array shaped ``(..., nx, ny, nz)``.
    perm : tuple[int, ...]
        New axis ``i`` is old axis ``perm[i]``.
    sign : numpy.ndarray
        ``(3,)`` of +1 or -1: -1 flips new axis ``i``.

    Returns
    -------
    numpy.ndarray
        The reoriented view of ``a``.
    """
    a = a.transpose(tuple(range(a.ndim - 3)) + tuple(a.ndim - 3 + p for p in perm))
    return a[(...,) + tuple(slice(None, None, int(s)) for s in sign)]


class PatchDataset(IterableDataset):
    """An endless stream of training patches from the runs of one split.

    Workers draw from generators seeded by ``(seed, worker id)``, so a seed and a worker count give the
    same patches. An epoch is a number of batches the training loop picks, not a pass over the data.

    Parameters
    ----------
    split : str
        The split to load: a value of the CSV's ``split`` column.
    csv : str or pathlib.Path
        The split CSV, one row per run.
    stores : str or pathlib.Path
        The folder of case stores.
    cache : str or pathlib.Path
        The folder of sampling indices. Builds the missing ones, which takes a few seconds per run.
    dataset : str, optional
        The dataset config, a name in configs/datasets/ or a YAML path.
    seed : int, optional
        The training run's seed.

    Raises
    ------
    ValueError
        If a run appears more than once in the CSV, which would put it in two splits.
    """

    def __init__(self, split, csv, stores, cache, dataset="ahmedml", seed=0):
        df = pd.read_csv(csv)
        if df.run.duplicated().any():
            raise ValueError(f"{csv}: runs in more than one split: {sorted(df.run[df.run.duplicated()])}")
        self.runs = df.run[df.split == split].tolist()
        self.seed = seed
        cfg, self.contract, self.rules = CaseConfig.load(dataset), Contract.load(), LabelRules.load()
        self.stores = {r: store_path(cfg, Path(stores), r) for r in self.runs}
        self.index = {r: load_index(s, r, Path(cache)) for r, s in self.stores.items()}
        scale = _load(configs_dir() / "contract_v1.yaml")["input_scales"]  # one for u, v and w: orientations swap them
        self.scale = np.array([scale["velocity"]] * 3 + [scale[c] for c in self.contract.channels[3:]], np.float32)

    def sample_centre(self, rng: np.random.Generator) -> tuple[int, str, np.ndarray]:
        """Draw a run, a kind of centre and a voxel.

        A run draws its kind with equal odds among those its index has voxels for, so a run with no
        vortex voxels gives that share to the other two.

        Parameters
        ----------
        rng : numpy.random.Generator
            The generator to draw from.

        Returns
        -------
        tuple[int, str, numpy.ndarray]
            The run number, the kind (a name in ``KINDS``) and the voxel, ``(3,)``.
        """
        run = self.runs[rng.integers(len(self.runs))]
        kinds = [k for k in KINDS if len(self.index[run][k])]
        kind = kinds[rng.integers(len(kinds))]
        return run, kind, self.index[run][kind][rng.integers(len(self.index[run][kind]))]

    def sample_patch(self, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        """Read one patch around a drawn centre and give it a random orientation.

        Parameters
        ----------
        rng : numpy.random.Generator
            The generator to draw from.

        Returns
        -------
        dict[str, torch.Tensor]
            ``inputs`` ``(6, 96, 96, 96)`` float32, scaled; ``labels`` ``(2, 96, 96, 96)`` uint8; and
            ``core`` ``(96, 96, 96)`` bool, where the labels count.
        """
        run, _, centre = self.sample_centre(rng)
        store = zarr.open_group(self.stores[run], mode="r")
        # one extra voxel on every side, so the erosion is right at the patch's faces
        lo = centre - PATCH // 2 - 1
        x, inside = read_crop(store["inputs"], lo, PATCH + 2)
        labels, _ = read_crop(store["labels"], lo, PATCH + 2)
        h_H = (1 - x[5]) * self.contract.ground_proximity_range_H  # exact where it matters: the floor is below the range
        _, core = core_mask((x[4] < 1) & inside, h_H, self.rules, self.contract.n_per_H)

        inner = (...,) + (slice(1, -1),) * 3
        x, labels, core = x[inner] / self.scale[:, None, None, None], labels[inner], core[inner[1:]]
        perm, sign = ORIENTATIONS[rng.integers(len(ORIENTATIONS))]
        x = orient(x, perm, sign)
        x[:3] = x[:3][list(perm)] * sign[:, None, None, None]  # the components follow the axes
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in
                (("inputs", x), ("labels", orient(labels, perm, sign)), ("core", orient(core, perm, sign)))}

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            yield self.sample_patch(rng)
