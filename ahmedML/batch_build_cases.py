"""
Download AhmedML 'runs' from HF and build the training case.
This script overlaps the download of runs with the processing
of their VTU files.

Up to --workers builds run at the same time. A downloader thread fetches
runs one at a time, but only while fewer than workers + 1 runs are on disk
(one per running build + one prefetched), so downloader never gets further ahead
than builder. A slot is freed when a build finishes or a download fails.

Runs whose <out>/<tag>_labels.npz exist are skipped.

The raw volume files (*.vtu) are deleted after a successful build
unless --keep-vtu is given.

    python batch_build_cases.py --runs data/geo_parameters_all.csv --workers 3 --save-vti
"""
import os

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading
import queue
import time
import shutil

import pandas as pd

from huggingface_hub import hf_hub_download, DryRunFileInfo

FILES=["ahmed_{r}.stl", "geo_parameters_{r}.csv", "force_mom_{r}.csv", "volume_{r}.vtu"]

def download(run: int, data: str):
    for f in FILES:
        t0 = time.time()
        f = f.format(r=run)
        p = Path(data) / f"run_{run}" / f
        if p.exists() and p.stat().st_size > 0:
            continue
        part_tgt = hf_hub_download(
            repo_id="neashton/ahmedml",
            filename=f"run_{run}/{f}",
            local_dir=data,
            repo_type="dataset",)
        target = Path(part_tgt).replace(p)
        print(f"[dl ] {target}  {target.stat().st_size/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)

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
    if ok and not args.keep_vtu:
        v = Path(args.data) / f"run_{run}" / f"volume_{run}.vtu"
        if v.exists():
            v.unlink()
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated ids, or path to a csv with a 'run' column.")
    ap.add_argument("--data", default="data", help="Where the downloaded datasets go")
    ap.add_argument("--out", default="out2", help="Where the {tag}_labels.npz go")
    ap.add_argument("--n-per-H", type=int, default=32)
    ap.add_argument("--keep-vtu", action="store_true", default=False)
    ap.add_argument("--workers", type=int, default=1, help="number of builds to run at the same time")
    ap.add_argument("--save-vti", action="store_true", default=False, help="pass --save-vti to build_training_case.py")
    ap.add_argument("extra", nargs="*", help="extra args passed to build_training_case.py (after --")
    args = ap.parse_args()
    runs = (pd.read_csv(args.runs)["run"].astype(int).tolist() if args.runs.endswith(".csv")
        else [int(x) for x in args.runs.split(",")])
    todo = [r for r in runs if not Path(args.out, f"ahmed_run{r}_dx_H_by_{args.n_per_H}_labels.npz").exists()]
    print(f"{len(runs)} runs requested, {len(todo)} to do: {todo}", flush=True)

    # limits the runs on disk to one per worker + one prefetched
    slots = threading.Semaphore(args.workers + 1)
    q = queue.Queue()
    def downloader():
        for r in todo:
            slots.acquire()
            try:
                download(r, args.data)
                q.put(r)
            except Exception as e:
                print(f"[dl ] run_{r}: {e}", flush=True)
                q.put(-r)
        q.put(None)
    threading.Thread(target=downloader, daemon=True).start()
    Path(args.out).mkdir(exist_ok=True)

    results = {}
    lock = threading.Lock()
    def record(r, ok):
        with lock:
            results[r] = ok
            bad = [r for r, ok in results.items() if not ok]
            print(f"done: {sum(results.values())}/{len(results)} ok; failed: {bad}", flush=True)
    def worker(r):
        try:
            ok = build(r, args)
        except Exception as e:
            print(f"[run] run_{r}: {e}", flush=True)
            ok = False
        finally:
            slots.release()
        record(r, ok)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while (r := q.get()) is not None:
            if r < 0:
                slots.release()
                record(-r, False)
                continue
            pool.submit(worker, r)

if __name__ == "__main__":
    main()
