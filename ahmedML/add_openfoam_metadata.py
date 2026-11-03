"""
Add the OpenFOAM header comment to VTUs that fix_vtu.py already fixed, as field data.

The original volume_<N>.vtu files start with a comment such as
    <!-- case='run1' region='region0' time='79.9998' index='133333' -->
which VTK's reader drops. fix_vtu.py now keeps it as one-value string arrays OpenFOAM_<key> in the
field data; this script does the same for the files fixed before that, without downloading them
again: it fetches the first bytes of each original from Hugging Face (at a pinned revision) with
an HTTP range request.

For each run with a valid volume_<N>.fixed.json and no "openfoam" entry in it, it reads the fixed
file, adds the arrays, writes volume_<N>.vtu.tmp with the same settings, re-reads it and checks
that it equals the in-memory grid and differs from the old file only by the new arrays, renames it
over volume_<N>.vtu and updates the report (output size, summary, openfoam, timings).

    python add_openfoam_metadata.py --runs 1-500 --data /data/cfd-rgba-net-data/ahmedML/data --workers 3
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import urllib.request

from huggingface_hub import hf_hub_url

import fix_vtu as F

UPSTREAM_REVISION = "02688c727cdb8dc8678e28abc6bbbb7e93c5fa15"  # neashton/ahmedml the VTUs were fixed from


def fetch_header(run, revision=UPSTREAM_REVISION, n=1024, tries=5):
    url = hf_hub_url(F.REPO, f"run_{run}/volume_{run}.vtu", repo_type="dataset", revision=revision)
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}"}),
                                        timeout=60) as f:
                return F.parse_openfoam_header(f.read(n)), url
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def needs_update(run, data):
    p = Path(data) / f"run_{run}" / f"volume_{run}.fixed.json"
    return F.is_done(run, data) and "openfoam" not in json.loads(p.read_text())


def update(run, data, meta, url):
    T = {}
    t0 = time.time()
    def lap(name):
        nonlocal t0
        T[name] = round(time.time() - t0, 2)
        t0 = time.time()

    d = Path(data) / f"run_{run}"
    dst, tmp, rep_path = d / f"volume_{run}.vtu", d / f"volume_{run}.vtu.tmp", d / f"volume_{run}.fixed.json"
    rep = json.loads(rep_path.read_text())
    # A few upstream cases carry a note, e.g. 'run107-problemrunning', 'run435-rerun-needed'; kept verbatim.
    case = (meta or {}).get("case", "")
    if case != f"run{run}" and not case.startswith(f"run{run}-"):
        raise RuntimeError(f"unexpected header for run {run}: {meta}")

    grid = F.read_vtu(dst)
    lap("read")
    before = F.summarize(grid)
    if F.diff_summaries(rep["summary"], before):
        raise RuntimeError(f"{dst} differs from its report in {F.diff_summaries(rep['summary'], before)}")
    lap("summarize")
    existing = F.openfoam_field_data(grid)
    if existing and existing != meta:
        raise RuntimeError(f"{dst} already has different OpenFOAM field data {existing}")
    F.add_openfoam_field_data(grid, meta)
    tagged = F.summarize(grid)
    new = [f"field_data.{F.OPENFOAM_PREFIX}{k}" for k in meta]
    changed = F.diff_summaries(before, tagged)
    if sorted(changed) != sorted(c for c in new if c not in [f"field_data.{F.OPENFOAM_PREFIX}{k}" for k in existing]):
        raise RuntimeError(f"adding the header changed {changed}")
    lap("tag")

    setting = rep["output"]["writer"]["setting"]
    F.write_vtu(grid, tmp, setting)
    lap("write")
    del grid
    reread = F.read_vtu(tmp)
    after = F.summarize(reread)
    if F.diff_summaries(tagged, after):
        raise RuntimeError(f"re-read {tmp} differs in {F.diff_summaries(tagged, after)}")
    if F.openfoam_field_data(reread) != meta:
        raise RuntimeError(f"re-read {tmp} has OpenFOAM field data {F.openfoam_field_data(reread)}")
    del reread
    lap("reread_compare")
    os.replace(tmp, dst)

    rep["openfoam"] = meta
    rep["openfoam_source"] = dict(url=url, revision=UPSTREAM_REVISION, added=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                  timings_s=T)
    rep["output"]["bytes"] = dst.stat().st_size
    rep["summary"] = after
    rtmp = rep_path.with_suffix(".json.tmp")
    rtmp.write_text(json.dumps(rep, indent=2))
    os.replace(rtmp, rep_path)
    return T


def _job(run, data, meta, url):
    try:
        return update(run, data, meta, url), None
    except Exception:
        return None, traceback.format_exc()


def parse_runs(s):
    if s.endswith(".csv"):
        import pandas as pd
        return pd.read_csv(s)["run"].astype(int).tolist()
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="e.g. 1-500, comma-separated ids/ranges, or a csv with a 'run' column")
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    runs = parse_runs(args.runs)
    unfixed = [r for r in runs if not F.is_done(r, args.data)]
    todo = [r for r in runs if r not in unfixed and needs_update(r, args.data)]
    print(f"{len(runs)} runs, {len(todo)} to tag, {len(unfixed)} not fixed (skipped): {unfixed}", flush=True)

    headers = {}
    for r in todo:
        headers[r] = fetch_header(r)
    print(f"fetched {len(headers)} headers", flush=True)

    ok, bad = 0, []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, max_tasks_per_child=1) as pool:
        futs = {pool.submit(_job, r, args.data, *headers[r]): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            T, err = fut.result()
            if err:
                bad.append(r)
                print(f"[tag] run_{r}: FAILED\n{err}", flush=True)
            else:
                ok += 1
                print(f"[tag] run_{r}: ok {headers[r][0]} {T}", flush=True)
            n = ok + len(bad)
            eta = (time.time() - t0) / n * (len(todo) - n) / 3600
            print(f"done: {ok}/{n} ok of {len(todo)}; failed: {sorted(bad)}; eta {eta:.1f} h", flush=True)
    print(f"finished: {ok} ok, {len(bad)} failed {sorted(bad)}", flush=True)


if __name__ == "__main__":
    main()
