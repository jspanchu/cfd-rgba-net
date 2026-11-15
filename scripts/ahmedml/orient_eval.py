"""
Score checkpoints on re-oriented validation cases: does the net still work when the flow runs along another
axis, or the floor lies on another side? Training augments with the 48 axis permutations and flips of a
cube. A net that was trained without them scores well on the stored orientation and poorly on the rest.

    python -I scripts/ahmedml/orient_eval.py --stores /data/cfd-rgba-net-data/ahmedML/out_v1 \\
        --out results/orient.json runs/r0_seed1/last.pt runs/r2_noaug/last.pt

Each case's inputs, labels and core are re-oriented together, and the velocity components follow the axes
(as in ``loader.PatchDataset.sample_patch``), so the reference stays exact. Scores come from ``evaluate.predict``
as it is now, with the zero margin and bfloat16 windows: they differ from older runs of this script, which
tiled the windows from the grid's corner. All checkpoints must share the ``--levels`` and ``--norm`` settings.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from cfd_rgba_net.batch import store_path
from cfd_rgba_net.config import CaseConfig, Contract, LabelRules, configs_dir
from cfd_rgba_net.evaluate import metrics, pool, predict, score
from cfd_rgba_net.label import CLASS_NAMES, core_mask
from cfd_rgba_net.loader import input_scales, orient
from cfd_rgba_net.model import DerivedFeatures, UNet3D

# name: (perm, sign). New axis i is old axis perm[i], flipped where sign is -1. Each name says what the net
# sees relative to the stored case, whose x is the flow direction, y the span and z the height.
ORIENTATIONS = {
    "identity": ((0, 1, 2), (1, 1, 1)),
    "mirror spanwise (y)": ((0, 1, 2), (1, -1, 1)),
    "reverse flow (x)": ((0, 1, 2), (-1, 1, 1)),
    "floor on top (z)": ((0, 1, 2), (1, 1, -1)),
    "yaw 90 (swap x, y)": ((1, 0, 2), (1, 1, 1)),
    "floor on side (swap x, z)": ((2, 1, 0), (1, 1, 1)),
    "floor on side (swap y, z)": ((0, 2, 1), (1, 1, 1)),
    "cyclic permutation with flips": ((1, 2, 0), (-1, 1, -1)),
}


def reorient(x: np.ndarray, ref: np.ndarray, perm: tuple[int, ...], sign: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Re-orient a case's inputs and reference labels.

    Parameters
    ----------
    x : numpy.ndarray
        Inputs, ``(6, nx, ny, nz)``, with u, v and w in the first three channels.
    ref : numpy.ndarray
        Reference class index, ``(nx, ny, nz)``.
    perm : tuple[int, ...]
        New axis ``i`` is old axis ``perm[i]``.
    sign : tuple[int, ...]
        ``(3,)`` of +1 or -1: -1 flips new axis ``i``.

    Returns
    -------
    numpy.ndarray
        The inputs on the new axes, with the velocity components permuted and flipped to follow them.
    numpy.ndarray
        The reference on the new axes.
    """
    s = np.array(sign)
    out = orient(x, perm, s).copy()
    out[:3] = out[:3][list(perm)] * s[:, None, None, None]
    return np.ascontiguousarray(out), np.ascontiguousarray(orient(ref, perm, s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+", help="model weights, e.g. runs/r0_seed1/last.pt")
    ap.add_argument("--stores", type=Path, required=True, help="the folder of case stores")
    ap.add_argument("--out", type=Path, required=True, help="where the JSON scores go")
    ap.add_argument("--dataset", default="ahmedml", help="dataset config: a name in configs/datasets/ or a YAML path")
    ap.add_argument("--csv", type=Path, default=configs_dir() / "splits" / "ahmedml_v1.csv")
    ap.add_argument("--nruns", type=int, default=10, help="score the first N of the pilot's fixed 10 validation runs")
    ap.add_argument("--levels", type=int, default=5, help="resolution levels of the UNet")
    ap.add_argument("--derived", action="store_true", help="the checkpoints were trained with --derived")
    ap.add_argument("--prior", action="store_true", help="the checkpoints were trained with --prior")
    ap.add_argument("--norm", default="instance", choices=["instance", "group", "batch"], help="UNet normalisation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, contract, rules = CaseConfig.load(args.dataset), Contract.load(), LabelRules.load()
    df = pd.read_csv(args.csv)
    runs = df.run[df.split == "val"].tolist()[::5][:args.nruns]  # the pilot's fixed 10 of the 50
    scale = input_scales(contract)
    models = {}
    for path in args.checkpoints:
        features = DerivedFeatures.from_configs(contract, rules) if args.derived else None
        model = UNet3D(len(contract.channels), args.levels, args.norm, features, args.prior).to(device).eval()
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        models[str(path)] = model

    scores = {c: {o: [] for o in ORIENTATIONS} for c in models}
    start = time.time()
    for run in runs:
        store = zarr.open_group(store_path(ds, args.stores, run), mode="r")
        x0, labels = store["inputs"][:], store["labels"][:]
        ref0 = labels[0] + 2 * labels[1]
        for name, (perm, sign) in ORIENTATIONS.items():
            x, ref = reorient(x0, ref0, perm, sign)
            core = core_mask(x[4] < 1, (1 - x[5]) * contract.ground_proximity_range_H, rules, contract.n_per_H)[1]
            for ckpt, model in models.items():
                p = predict(model, x, scale, device) * torch.from_numpy(core).to(device)
                scores[ckpt][name].append(score(p.argmax(0).cpu().numpy(), ref, core, x[4], surface_dice=False))
        print(f"run {run} done, {time.time() - start:.0f} s", flush=True)

    summary = {c: {o: [round(metrics(pool(v)["all"][k])["iou"], 4) for k in CLASS_NAMES] for o, v in d.items()}
               for c, d in scores.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(runs=runs, classes=list(CLASS_NAMES), iou=summary), indent=1))
    print("pooled IoU, " + ", ".join(CLASS_NAMES))
    for c, d in summary.items():
        print(c)
        for o, iou in d.items():
            print(f"  {o:32s} " + "  ".join(f"{v:.3f}" for v in iou))


if __name__ == "__main__":
    main()
