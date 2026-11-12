"""
Compute the input scales of the train split (roadmap, Phase 1.1 task 2) and print them as YAML for
the ``input_scales`` entry of configs/contract_v1.yaml.

Over the fluid voxels (body_proximity < 1) of every train run: ``velocity`` is
sqrt(mean(u*^2 + v*^2 + w*^2) / 3), shared by u*, v* and w* so that the 48 orientations leave it unchanged;
Cp, body_proximity and ground_proximity each get their own standard deviation.

    python scripts/ahmedml/compute_scales.py --out /data/cfd-rgba-net-data/ahmedML/out_v1 \\
        --csv configs/splits/ahmedml_v1.csv
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def sums(store):
    """Sum the fluid voxels of one store, in float64.

    Parameters
    ----------
    store : pathlib.Path
        The case store.

    Returns
    -------
    numpy.ndarray
        ``(4, 3)``. Rows: u*^2 + v*^2 + w*^2, Cp, body_proximity, ground_proximity. Columns: sum,
        sum of squares, voxel count.
    """
    x = zarr.open_group(store, mode="r")["inputs"][:]
    x = x[:, x[4] < 1].astype(np.float64)
    rows = ((x[:3] ** 2).sum(0), *x[3:])
    return np.array([[v.sum(), (v ** 2).sum(), v.size] for v in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="folder of ahmedml_run<N>.zarr stores")
    ap.add_argument("--csv", required=True, help="split CSV")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    runs = pd.read_csv(args.csv).query("split == 'train'").run
    stores = [Path(args.out) / f"ahmedml_run{r}.zarr" for r in runs]
    total, t0 = 0, time.time()
    with Pool(args.workers) as pool:
        for i, t in enumerate(pool.imap_unordered(sums, stores), 1):
            total = total + t
            elapsed = time.time() - t0
            eta = elapsed / i * (len(stores) - i)
            print(f"{i}/{len(stores)}  {elapsed / 60:.1f} min elapsed, ETA {eta / 60:.1f} min", file=sys.stderr, flush=True)
    mean, mean_sq, n = total[:, 0] / total[:, 2], total[:, 1] / total[:, 2], total[:, 2]
    std = np.sqrt(mean_sq - mean ** 2)
    print(f"# {len(runs)} train runs, {int(n[0])} fluid voxels")
    print("input_scales:")
    print(f"  velocity: {np.sqrt(mean_sq[0] / 3):.6g}")
    for name, s in zip(("Cp", "body_proximity", "ground_proximity"), std[1:]):
        print(f"  {name}: {s:.6g}")


if __name__ == "__main__":
    main()
