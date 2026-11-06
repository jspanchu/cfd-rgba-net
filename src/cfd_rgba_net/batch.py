"""
Build many training cases of one dataset, downloading each run from Hugging Face as it is needed.

    cfd-rgba-net-batch --dataset windsorml --runs 0-349 --out /data/cfd-rgba-net-data/windsorML/out_v1
    cfd-rgba-net-batch --dataset ahmedml --runs ahmedML/data/geo_parameters_all.csv \
        --out /data/cfd-rgba-net-data/ahmedML/out_v1 --workers 6

For each run, the files named in the dataset config (`files`: volume, body, geo_parameters) are
downloaded from its `hf_repo` into <work>/run_<N>/ (default: a folder in the system temp directory,
not the --out disk), built with cfd_rgba_net.build_case in a subprocess, and deleted after a
successful build, so only the training data stays. After a failed build they are kept for diagnosis;
--keep keeps every download.

Up to --workers builds run at the same time, each with --threads VTK threads (default: all cores /
workers). Runs whose <out>/<dataset>_run<N>.zarr exists are skipped: a store is renamed into place
only when complete.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import CaseConfig


def parse_runs(spec: str) -> list[int]:
    """'1,5,7', '0-349', '1-3,10', or a CSV file with a 'run' column."""
    if spec.endswith(".csv"):
        import pandas as pd
        return pd.read_csv(spec)["run"].astype(int).tolist()
    runs: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        runs += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return runs


def download(cfg: CaseConfig, run: int, work: Path) -> None:
    from huggingface_hub import hf_hub_download
    if not cfg.hf_repo:
        raise ValueError(f"{cfg.dataset}: no hf_repo in the dataset config")
    for key, template in cfg.files.items():
        rel = template.format(run=run)
        target = work / rel
        if target.exists() and target.stat().st_size > 0:
            continue
        t0 = time.time()
        hf_hub_download(repo_id=cfg.hf_repo, filename=rel, local_dir=work, repo_type="dataset")
        print(f"[dl ] {cfg.hf_repo}: {rel}  {target.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s",
              flush=True)


def store_path(cfg: CaseConfig, out: Path, run: int) -> Path:
    return out / f"{cfg.dataset.lower()}_run{run}.zarr"


def build(cfg: CaseConfig, run: int, args) -> bool:
    log = args.out / f"log_{cfg.dataset.lower()}_run{run}.txt"
    print(f"[run] build run_{run} (log {log})", flush=True)
    cmd = [sys.executable, "-u", "-m", "cfd_rgba_net.build_case", "--dataset", args.dataset, "--run", str(run),
           "--data", str(args.work), "--out", str(args.out), "--threads", str(args.threads)]
    cmd += (["--save-vti"] if args.save_vti else []) + args.extra
    t0 = time.time()
    with log.open("w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    ok = r.returncode == 0 and store_path(cfg, args.out, run).exists()
    print(f"[run] run_{run}: {'ok' if ok else f'FAILED (exit {r.returncode})'} in {time.time() - t0:.0f}s", flush=True)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset config: a name in configs/datasets/ or a YAML path")
    ap.add_argument("--runs", required=True, help="'1,5,7', '0-349', or a CSV file with a 'run' column")
    ap.add_argument("--out", type=Path, required=True, help="where the <dataset>_run<N>.zarr stores and logs go")
    ap.add_argument("--work", type=Path, default=None,
                    help="where run files are downloaded (default: <temp>/cfd-rgba-net-<dataset>)")
    ap.add_argument("--keep", action="store_true", help="keep the downloaded run files")
    ap.add_argument("--workers", type=int, default=1, help="builds at the same time")
    ap.add_argument("--threads", type=int, default=None, help="VTK threads per build (default: cores / workers)")
    ap.add_argument("--save-vti", action="store_true", help="pass --save-vti to the builder")
    ap.add_argument("extra", nargs="*", help="extra args passed to cfd_rgba_net.build_case (after --)")
    args = ap.parse_args(argv)
    cfg = CaseConfig.load(args.dataset)
    args.work = args.work or Path(tempfile.gettempdir()) / f"cfd-rgba-net-{cfg.dataset.lower()}"
    args.threads = args.threads or max(1, (os.cpu_count() or 1) // args.workers)
    runs = parse_runs(args.runs)
    todo = [r for r in runs if not store_path(cfg, args.out, r).exists()]
    print(f"{cfg.dataset}: {len(runs)} runs requested, {len(todo)} to do; downloads in {args.work}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[int, bool] = {}
    lock = threading.Lock()

    def worker(r: int):
        try:
            download(cfg, r, args.work)
            ok = build(cfg, r, args)
        except Exception as e:  # noqa: BLE001  (report and go on with the next run)
            print(f"[run] run_{r}: {e}", flush=True)
            ok = False
        if ok and not args.keep:
            shutil.rmtree(args.work / f"run_{r}", ignore_errors=True)
        with lock:
            results[r] = ok
            bad = sorted(k for k, v in results.items() if not v)
            print(f"done: {sum(results.values())}/{len(results)} ok; failed: {bad}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, todo))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
