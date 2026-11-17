"""
Add the net's ``features`` array to case stores built before ``build_case`` wrote it.

The features are lambda2*, Q* and |omega|* from the grid's own velocity (``baseline.grid_features``), computed
from the stored ``inputs`` exactly as ``build_case`` computes them, so an updated store matches a fresh build.
Stores that already have ``features`` are skipped, so the script can be rerun after a crash.

    python scripts/ahmedml/add_features.py --stores /data/cfd-rgba-net-data/ahmedML/out_v1 --workers 8

Each store's array is written as ``features.tmp`` and renamed to ``features`` once complete. Only the new
array's folder is written: the root ``zarr.json``, whose time stamps the cached sampling indices, is untouched.
"""
import os

# one thread per worker for NumPy's eigenvalue solves; must be set before NumPy is first imported
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import zarr  # noqa: E402

from cfd_rgba_net.baseline import grid_features  # noqa: E402
from cfd_rgba_net.config import Contract  # noqa: E402
from cfd_rgba_net.io import _put  # noqa: E402


def add_features(store: Path) -> str:
    """Compute one store's features and write them as its ``features`` array.

    Parameters
    ----------
    store : pathlib.Path
        The case store.

    Returns
    -------
    str
        ``"added"``, or ``"skipped"`` when the store already has ``features``.
    """
    if (store / "features").exists():
        return "skipped"
    shutil.rmtree(store / "features.tmp", ignore_errors=True)  # left by a crash
    group = zarr.open_group(store, mode="r+")
    x = group["inputs"][:]
    _put(group, "features.tmp", grid_features(x[:3], x[4] < 1, 1 / Contract.load().n_per_H))
    (store / "features.tmp").rename(store / "features")
    return "added"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stores", type=Path, required=True, help="folder of <dataset>_run<N>.zarr stores")
    ap.add_argument("--workers", type=int, default=8, help="stores processed at once, about 4 GB of memory each")
    args = ap.parse_args()
    stores = sorted(args.stores.glob("*.zarr"))
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (store, status) in enumerate(zip(stores, pool.imap(add_features, stores)), 1):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(stores) - i)
            print(f"{i}/{len(stores)} {store.name} {status}  {elapsed / 60:.1f} min elapsed, ETA {eta / 60:.1f} min",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
