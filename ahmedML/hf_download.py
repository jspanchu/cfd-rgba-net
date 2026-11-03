"""Download AhmedML run files from Hugging Face.

The default repo is jspanchu/ahmedml, a copy of neashton/ahmedml whose volume_<N>.vtu files have the
VTK_WEDGE point order fixed (see fix_vtu.py). Only fix_vtu.py downloads from the upstream repo.
"""
from pathlib import Path
import time

from huggingface_hub import hf_hub_download

FIXED_REPO = "jspanchu/ahmedml"
FILES = ["ahmed_{r}.stl", "geo_parameters_{r}.csv", "force_mom_{r}.csv", "volume_{r}.vtu"]


def download(run: int, data: str | Path, files: list[str] = FILES, repo: str = FIXED_REPO):
    """Fetch files (templates in {r}) of run from repo into <data>/run_<run>/, skipping non-empty existing ones."""
    for f in files:
        t0 = time.time()
        f = f.format(r=run)
        p = Path(data) / f"run_{run}" / f
        if p.exists() and p.stat().st_size > 0:
            continue
        part_tgt = hf_hub_download(
            repo_id=repo,
            filename=f"run_{run}/{f}",
            local_dir=data,
            repo_type="dataset",)
        target = Path(part_tgt).replace(p)
        print(f"[dl ] {repo}: {target}  {target.stat().st_size/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
