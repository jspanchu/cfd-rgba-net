"""
Build the training cases of AhmedML 'runs' from the fixed volume files.

The run files (volume_<N>.vtu, STL, CSVs) are downloaded into <data>/run_<N>/ from jspanchu/ahmedml,
the copy of the dataset with the VTK_WEDGE point order fixed, unless they are already there. They
are kept after the build.

Up to --workers builds run at the same time. Runs whose <out>/<tag>_labels.npz exist are skipped.

    python batch_build_cases.py --runs data/geo_parameters_all.csv --data /data/cfd-rgba-net-data/ahmedML/data \
        --workers 3 --save-vti
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading
import time

import pandas as pd

from hf_download import FIXED_REPO, download

def build(run: int, args):
    tag = f"ahmed_run{run}_dx_H_by_{args.n_per_H}"
    log = Path(args.out) / f"log_run{run}_dx_H_by_{args.n_per_H}.txt"
    print(f"[run] build run_{run} (log {log})", flush=True)
    cmd = [sys.executable, "-u", Path(__file__).with_name("build_training_case.py"), "--run", str(run), "--data",
        args.data, "--out", args.out, "--n-per-H", str(args.n_per_H)] + (["--save-vti"] if args.save_vti else []) + args.extra
    t0 = time.time()
    with log.open(mode="w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    ok = r.returncode == 0 and Path(args.out, f"{tag}_labels.npz").exists()
    print(f"[run] run_{run}: {'ok' if ok else 'FAILED'} in {time.time()-t0:.0f}s (log {log})", flush=True)
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated ids, or path to a csv with a 'run' column.")
    ap.add_argument("--data", default="data", help=f"dataset root holding run_<N>/; missing files are downloaded from {FIXED_REPO}")
    ap.add_argument("--out", default="out2", help="Where the {tag}_labels.npz go")
    ap.add_argument("--n-per-H", type=int, default=32)
    ap.add_argument("--workers", type=int, default=1, help="number of builds to run at the same time")
    ap.add_argument("--save-vti", action="store_true", default=False, help="pass --save-vti to build_training_case.py")
    ap.add_argument("extra", nargs="*", help="extra args passed to build_training_case.py (after --")
    args = ap.parse_args()
    runs = (pd.read_csv(args.runs)["run"].astype(int).tolist() if args.runs.endswith(".csv")
        else [int(x) for x in args.runs.split(",")])
    todo = [r for r in runs if not Path(args.out, f"ahmed_run{r}_dx_H_by_{args.n_per_H}_labels.npz").exists()]
    print(f"{len(runs)} runs requested, {len(todo)} to do: {todo}", flush=True)
    Path(args.out).mkdir(exist_ok=True)

    results = {}
    lock = threading.Lock()
    def worker(r):
        try:
            download(r, args.data)
            ok = build(r, args)
        except Exception as e:
            print(f"[run] run_{r}: {e}", flush=True)
            ok = False
        with lock:
            results[r] = ok
            bad = [r for r, ok in results.items() if not ok]
            print(f"done: {sum(results.values())}/{len(results)} ok; failed: {bad}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pool.map(worker, todo)

if __name__ == "__main__":
    main()
