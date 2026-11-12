"""
Build and cache the sampling index of every run of a split (roadmap, Phase 1.1 task 2), in parallel, with
progress. Runs whose cached index is current are skipped, so it can be stopped and restarted.

    python scripts/ahmedml/build_index.py --out /data/cfd-rgba-net-data/ahmedML/out_v1 \\
        --csv configs/splits/ahmedml_v1.csv --cache /data/cfd-rgba-net-data/ahmedML/index
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from cfd_rgba_net.loader import load_index


def build(job):
    """Build one run's index, or find it current in the cache.

    Parameters
    ----------
    job : tuple[pathlib.Path, int, pathlib.Path]
        The run's store, the run number and the cache folder.
    """
    load_index(*job)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="folder of ahmedml_run<N>.zarr stores")
    ap.add_argument("--csv", required=True, help="split CSV")
    ap.add_argument("--cache", required=True, help="folder of the cached indices")
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    runs = pd.read_csv(args.csv).query("split == @args.split").run
    jobs = [(Path(args.out) / f"ahmedml_run{r}.zarr", r, Path(args.cache)) for r in runs]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(build, jobs), 1):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(jobs) - i)
            print(f"{i}/{len(jobs)}  {elapsed / 60:.1f} min elapsed, ETA {eta / 60:.1f} min", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
