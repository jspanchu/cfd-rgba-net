"""
Assign every AhmedML run (1 to 499) to one split and write the result as a CSV (roadmap, Phase 1.1 task 1).

Splits (made once, committed, never changed):

    test_extrap   slant below 10 or above 65 degrees (31 runs); never trained on
    train         368 runs
    val           50 runs
    test_interp   50 runs

The other 468 runs are tagged vortex-rich (vortex fraction above 0.1% of the fluid voxels) or vortex-poor.
Within each tag the runs are sorted by slant and val and test runs are picked at evenly spaced ranks,
so every split covers the slant range and holds the same share of vortex-rich runs. The three runs that
tuned the baseline thresholds (349, 428, 325) are always train. Run 44 is in the extrapolation band
and tuned the baseline too, so it is not scored. Run 500 is left out: its geometry parameters are a copy
of run 499's.

The training subsets of 50, 100 and 200 runs are nested and spread over slant the same way: each is
picked from the previous larger one, per tag.

Slant and vortex fraction come from the case stores' attributes (``geometry.geo_parameters`` is the run's
geo_parameters_<N>.csv).

    python scripts/ahmedml/make_splits.py --out /data/cfd-rgba-net-data/ahmedML/out_v1 \\
        --csv configs/splits/ahmedml_v1.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

RUNS = range(1, 500)
SLANT_RANGE = (10.0, 65.0)  # degrees; runs outside are the extrapolation test
VORTEX_RICH = 1e-3  # fraction of the fluid voxels
N_EVAL = 50  # runs in each of val and test_interp
SUBSETS = (200, 100, 50)  # nested training subsets, largest first
TRAIN_RUNS = (349, 428, 325)  # tuned the baseline thresholds
NOT_SCORED = (44,)


def spread(sorted_runs, k, offset=0.5):
    """Pick k evenly spaced entries of a list.

    Parameters
    ----------
    sorted_runs : list
        The runs, sorted by slant.
    k : int
        The number of runs to pick.
    offset : float
        Where in each of the k equal stretches to pick, from 0 (first) to 1 (last).

    Returns
    -------
    list
        The k picked runs, still sorted by slant.
    """
    return [sorted_runs[int((i + offset) * len(sorted_runs) / k)] for i in range(k)]


def read_runs(out):
    """Read slant and vortex fraction of every run from its store.

    Parameters
    ----------
    out : str or pathlib.Path
        The folder of ``ahmedml_run<N>.zarr`` stores.

    Returns
    -------
    pandas.DataFrame
        One row per run with columns ``slant`` and ``vortex``, indexed by run.
    """
    rows = {}
    for run in RUNS:
        attrs = zarr.open_group(Path(out) / f"ahmedml_run{run}.zarr", mode="r").attrs
        rows[run] = {
            "slant": attrs["geometry"]["geo_parameters"]["slant-angle-degrees"],
            "vortex": attrs["class_fractions_fluid"]["vortex"],
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("run")


def make_splits(df):
    """Assign each run a regime, a split and the smallest training subset it belongs to.

    Parameters
    ----------
    df : pandas.DataFrame
        The runs, indexed by run, with columns ``slant`` and ``vortex``.

    Returns
    -------
    pandas.DataFrame
        ``df`` plus the columns ``regime``, ``split``, ``scored`` and ``subset``. ``subset`` is the size
        of the smallest training subset holding the run (the train size for a run in none), empty outside train.
    """
    df = df.copy()
    df["regime"] = np.where(df.vortex > VORTEX_RICH, "vortex-rich", "vortex-poor")
    df["scored"] = ~df.index.isin(NOT_SCORED)
    extrap = ~df.slant.between(*SLANT_RANGE)
    df["split"] = np.where(extrap, "test_extrap", "train")

    def by_regime(runs, size, offset):
        """Pick ``size`` runs, spread over slant within each regime in proportion to its share."""
        sub = df.loc[runs].sort_values("slant")
        picks = [
            spread(g.index.tolist(), round(size * len(g) / len(sub)), offset) for _, g in sub.groupby("regime")
        ]
        return sorted(run for p in picks for run in p)

    # val and test come from the runs that are not forced into train
    free = df.index[~extrap & ~df.index.isin(TRAIN_RUNS)]
    df.loc[by_regime(free, N_EVAL, 0.25), "split"] = "val"
    df.loc[by_regime(free, N_EVAL, 0.75), "split"] = "test_interp"

    members = df.index[df.split == "train"]
    df["subset"] = pd.Series(len(members), index=members, dtype="Int64")
    for size in SUBSETS:
        members = by_regime(members, size, 0.5)
        df.loc[members, "subset"] = size
    df["subset"] = df.subset.astype("Int64")  # integers, not floats, in the CSV
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="folder of ahmedml_run<N>.zarr stores")
    ap.add_argument("--csv", required=True, help="split CSV to write")
    args = ap.parse_args()
    df = make_splits(read_runs(args.out))
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df[["slant", "vortex", "regime", "split", "scored", "subset"]].to_csv(args.csv)
    print(df.groupby(["split", "regime"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
