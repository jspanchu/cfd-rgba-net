r"""
Build many training cases of one dataset, downloading each run from Hugging Face
as it is needed.

    cfd-rgba-net-batch --dataset windsorml --runs 0-349 \
        --out /data/cfd-rgba-net-data/windsorML/out_v1
    cfd-rgba-net-batch --dataset ahmedml \
        --runs ahmedML/data/geo_parameters_all.csv \
        --out /data/cfd-rgba-net-data/ahmedML/out_v1 --workers 6

For each run, the files named in the dataset config (`files`: volume, body,
geo_parameters) are downloaded from its `hf_repo` into <work>/run_<N>/ (default:
a folder in the system temp directory, not the --out disk), built with
cfd_rgba_net.build_case in a subprocess, and deleted after a successful build,
so only the training data stays. After a failed build they are kept for
diagnosis; --keep keeps every download.

Up to --workers builds run at the same time, each with --threads threads
(default: all cores / workers). A separate download queue fetches the next runs
while the builds run, up to --prefetch runs ahead of the builders, so the GPU
does not wait for the network. Runs whose <out>/<dataset>_run<N>.zarr exists
are skipped: a store is renamed into place only when complete. A store built
with an older config or contract is skipped too.
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .build_case import store_path
from .config import CaseConfig


def parse_runs(spec: str) -> list[int]:
    """Parse a specification of run numbers.

    Parameters
    ----------
    spec : str
        Comma-separated run numbers and inclusive ranges, such as ``'1,5,7'``,
        ``'0-349'`` or ``'1-3,10'``; or the path of a CSV file (ending in
        ``.csv``) with a ``run`` column.

    Returns
    -------
    list[int]
        The run numbers in the order given, each once.

    Raises
    ------
    ValueError
        If a range runs backwards (``'5-3'``) or an entry is not a number.
    """
    if spec.endswith(".csv"):
        import pandas as pd

        return list(
            dict.fromkeys(pd.read_csv(spec)["run"].astype(int).tolist())
        )
    runs: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        if b:
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"run range {part.strip()!r} runs backwards")
            runs += range(lo, hi + 1)
        else:
            runs.append(int(a))
    return list(dict.fromkeys(runs))


def download(cfg: CaseConfig, run: int, work: Path) -> None:
    """Download the files of one run from Hugging Face.

    Files that already exist in ``work`` with a non-zero size are skipped.

    Parameters
    ----------
    cfg : CaseConfig
        The dataset configuration. Supplies ``hf_repo`` and the file paths.
    run : int
        The run number, which replaces ``{run}`` in the file paths.
    work : pathlib.Path
        The directory to download into. Each file lands at its path from the
        config, relative to it.

    Raises
    ------
    ValueError
        If the dataset config has no ``hf_repo``.
    """
    from huggingface_hub import hf_hub_download

    if not cfg.hf_repo:
        raise ValueError(f"{cfg.dataset}: no hf_repo in the dataset config")
    for template in cfg.files.values():
        rel = template.format(run=run)
        target = work / rel
        if target.exists() and target.stat().st_size > 0:
            continue
        t0 = time.time()
        hf_hub_download(
            repo_id=cfg.hf_repo,
            filename=rel,
            local_dir=work,
            repo_type="dataset",
        )
        print(
            f"[dl ] {cfg.hf_repo}: {rel}  "
            f"{target.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s",
            flush=True,
        )


def build(cfg: CaseConfig, run: int, args: argparse.Namespace) -> bool:
    """Build one run with ``cfd_rgba_net.build_case`` in a subprocess.

    The subprocess's output goes to ``<out>/log_<dataset>_run<run>.txt``, which
    an earlier run's log replaces.

    Parameters
    ----------
    cfg : CaseConfig
        The dataset configuration.
    run : int
        The run number.
    args : argparse.Namespace
        The parsed command line, as completed by ``main``. Uses ``dataset``
        (str), ``work`` (the data root, pathlib.Path), ``out`` (pathlib.Path),
        ``threads`` (int), ``save_vti`` (bool) and ``extra`` (list[str] of
        further ``build_case`` arguments).

    Returns
    -------
    bool
        True if the subprocess exited with 0 and the run's store exists.
    """
    log = args.out / f"log_{cfg.case_tag(run)}.txt"
    print(f"[run] build run_{run} (log {log})", flush=True)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "cfd_rgba_net.build_case",
        "--dataset",
        args.dataset,
        "--run",
        str(run),
        "--data",
        str(args.work),
        "--out",
        str(args.out),
        "--threads",
        str(args.threads),
    ]
    cmd += (["--save-vti"] if args.save_vti else []) + args.extra
    t0 = time.time()
    with log.open("w") as fh:
        r = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, check=False
        )
    ok = r.returncode == 0 and store_path(cfg, args.out, run).exists()
    status = "ok" if ok else f"FAILED (exit {r.returncode})"
    print(f"[run] run_{run}: {status} in {time.time() - t0:.0f}s", flush=True)
    return ok


def main(argv: list[str] | None = None) -> int:
    """Run the ``cfd-rgba-net-batch`` command line.

    Skips the runs whose store exists, then downloads and builds the others,
    ``--workers`` at a time. A failure is reported and does not stop the other
    runs. The downloads of a run that built are deleted unless ``--keep`` is
    given; those of a failed run are kept.

    Parameters
    ----------
    argv : list[str] or None, optional
        The arguments. ``None`` means ``sys.argv[1:]``.

    Returns
    -------
    int
        The exit code: 0 if every run built, 1 otherwise.
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="dataset config: a name in configs/datasets/ or a YAML path",
    )
    ap.add_argument(
        "--runs",
        required=True,
        help="'1,5,7', '0-349', or a CSV file with a 'run' column",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where the <dataset>_run<N>.zarr stores and logs go",
    )
    ap.add_argument(
        "--work",
        type=Path,
        default=None,
        help="where run files are downloaded "
        "(default: <temp>/cfd-rgba-net-<dataset>)",
    )
    ap.add_argument(
        "--keep", action="store_true", help="keep the downloaded run files"
    )
    ap.add_argument(
        "--workers", type=int, default=1, help="builds at the same time"
    )
    ap.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="runs downloaded ahead of the builders; at most workers + "
        "prefetch runs are on disk at once",
    )
    ap.add_argument(
        "--downloads",
        type=int,
        default=1,
        help="downloads at the same time",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="threads per build, for VTK, OpenMP, MKL and OpenBLAS "
        "(default: cores / workers)",
    )
    ap.add_argument(
        "--save-vti", action="store_true", help="pass --save-vti to the builder"
    )
    ap.add_argument(
        "extra",
        nargs="*",
        help="extra args passed to cfd_rgba_net.build_case (after --)",
    )
    args = ap.parse_args(argv)
    cfg = CaseConfig.load(args.dataset)
    args.work = (
        args.work
        or Path(tempfile.gettempdir()) / f"cfd-rgba-net-{cfg.dataset.lower()}"
    )
    args.threads = args.threads or max(1, (os.cpu_count() or 1) // args.workers)
    runs = parse_runs(args.runs)
    todo = [r for r in runs if not store_path(cfg, args.out, r).exists()]
    print(
        f"{cfg.dataset}: {len(runs)} runs requested, {len(todo)} to do; "
        f"downloads in {args.work}",
        flush=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[int, bool] = {}
    lock = threading.Lock()

    # runs on disk (downloading, downloaded or building): bounds the disk use
    on_disk = threading.Semaphore(args.workers + args.prefetch)
    ready: queue.Queue = queue.Queue()

    def fetch(r: int):
        try:
            download(cfg, r, args.work)
            ready.put((r, None))
        except Exception as e:  # noqa: BLE001  (reported by the builder)
            ready.put((r, e))

    def feed():
        with ThreadPoolExecutor(max_workers=args.downloads) as dl_pool:
            for r in todo:
                on_disk.acquire()
                dl_pool.submit(fetch, r)
        for _ in range(args.workers):
            ready.put(None)

    def builder():
        while (item := ready.get()) is not None:
            r, err = item
            try:
                if err is not None:
                    raise err
                ok = build(cfg, r, args)
            except Exception as e:  # noqa: BLE001  (report and go on with the next run)
                print(f"[run] run_{r}: {e}", flush=True)
                ok = False
            if ok and not args.keep:
                shutil.rmtree(args.work / f"run_{r}", ignore_errors=True)
            on_disk.release()
            with lock:
                results[r] = ok
                bad = sorted(k for k, v in results.items() if not v)
                print(
                    f"done: {sum(results.values())}/{len(results)} ok; "
                    f"failed: {bad}",
                    flush=True,
                )

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in range(args.workers):
            pool.submit(builder)
    feeder.join()
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
