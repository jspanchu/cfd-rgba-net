r"""
Evaluation harness (roadmap, Phase 1.1 task 6): turns a checkpoint and a split into scores.

    cfd-rgba-net-evaluate --checkpoint runs/pilot/best.pt --split val --stores /data/.../out_v1 \
        --out results/pilot_val --baseline

Sliding-window inference runs the net on whole cases. Metrics compare a mask with the reference labels per
class over core voxels, in three regions. The baseline's masks go through the same metrics. One JSON file
per run and one per split are written to ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
import zarr

from .baseline import baseline_classes
from .batch import parse_runs, store_path
from .config import CaseConfig, Contract, LabelRules, configs_dir
from .io import git_state
from .label import CLASS_NAMES, core_mask
from .loader import PATCH
from .model import UNet3D

STEP = PATCH // 2  # windows overlap by half
# Zero margin around the case. Without it the grid's low edges, above all the floor, sit on a window's edge, where
# the net lacks the zero context that training patches at the grid edge have. On the 10 fixed validation runs, a
# net trained with the pilot's settings scored a far-field shear-layer IoU of 0.05 without the margin and 0.42
# with it. Half a stride keeps every grid edge away from the window edges.
PAD = STEP // 2
WINDOW_BATCH = 8  # windows per forward pass
NEAR_BODY = 0.9375  # body_proximity of 0.25 H from the body: 1 - 0.25 / 4
COUNTS = ("inter", "pred", "ref", "surf_match", "surf")


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run the net over a whole case with overlapping windows.

    The case gets a zero margin of ``PAD`` voxels on every side, and windows are zero-padded where they leave
    that. Their predictions are blended with a Gaussian weight (sigma = 1/8 of the window), so a window's centre
    counts more than its edges. The margin is cut off again before the function returns. On a GPU the windows
    run in bfloat16, as the training steps do.

    Parameters
    ----------
    model : torch.nn.Module
        The network in eval mode, on ``device``: it returns logits ``(1, 3, D, H, W)``.
    x : torch.Tensor
        The case's raw float32 ``features``, ``(3, nx, ny, nz)``. The padded copy stays on ``x``'s device,
        and each batch of windows moves to ``device``.
    device : torch.device
        Where the windows run and their predictions are blended. Blending on a GPU takes about 12 bytes per
        voxel of the padded case there, which is about 1 GiB for an AhmedML case.

    Returns
    -------
    torch.Tensor
        Float32 probabilities of none, vortex and shear layer, ``(3, nx, ny, nz)``, on ``device``.
    """
    dims = x.shape[1:]
    inner = (slice(None),) + tuple(slice(PAD, PAD + n) for n in dims)  # the case within the padded volume
    size = [PATCH + STEP * -(-max(n + 2 * PAD - PATCH, 0) // STEP) for n in dims]  # sides the windows tile exactly
    xp = x.new_zeros(x.shape[0], *size)
    xp[inner] = x

    g = torch.arange(PATCH) - (PATCH - 1) / 2
    g = torch.exp(-g**2 / (2 * (PATCH / 8) ** 2))
    weight = (g[:, None, None] * g[None, :, None] * g[None, None, :]).to(device)
    total, norm = torch.zeros(3, *size, device=device), torch.zeros(*size, device=device)
    wins = [tuple(slice(c, c + PATCH) for c in corner)
            for corner in itertools.product(*(range(0, n - PATCH + 1, STEP) for n in size))]
    for batch in itertools.batched(wins, WINDOW_BATCH):
        inputs = torch.stack([xp[(slice(None),) + win] for win in batch]).to(device)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inputs)
        for win, p in zip(batch, logits.float().softmax(1)):
            total[(slice(None),) + win] += p * weight
            norm[win] += weight
    return (total / norm)[inner]


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask)


def class_counts(pred: np.ndarray, ref: np.ndarray, region: np.ndarray, surface_dice: bool = True) -> dict[str, int]:
    """Count what the metrics need for one class in one region.

    Parameters
    ----------
    pred : numpy.ndarray
        Bool mask of the class from the net or the baseline, ``(nx, ny, nz)``.
    ref : numpy.ndarray
        Bool mask of the class in the reference labels.
    region : numpy.ndarray
        Bool mask of the voxels that count.
    surface_dice : bool, optional
        False skips the surface counts, which take about half the scoring time; they read 0.

    Returns
    -------
    dict[str, int]
        ``inter``, ``pred`` and ``ref`` (voxel counts), and ``surf_match`` and ``surf`` for the surface
        Dice at a tolerance of 1 voxel: the boundary voxels of each mask lying within 1 voxel of the
        other's boundary, and all boundary voxels. Counts add up across runs.
    """
    pred, ref = pred & region, ref & region
    counts = dict(inter=int((pred & ref).sum()), pred=int(pred.sum()), ref=int(ref.sum()), surf_match=0, surf=0)
    if surface_dice:
        sp, sr = surface(pred), surface(ref)
        counts["surf_match"] = int((sp & ndi.binary_dilation(sr)).sum() + (sr & ndi.binary_dilation(sp)).sum())
        counts["surf"] = int(sp.sum() + sr.sum())
    return counts


def metrics(c: dict[str, int]) -> dict[str, float | None]:
    """Turn counts into IoU, Dice and surface Dice.

    Parameters
    ----------
    c : dict[str, int]
        Counts as ``class_counts`` returns, of one run or summed over runs.

    Returns
    -------
    dict[str, float or None]
        ``iou``, ``dice`` and ``surface_dice``; None where the denominator is 0.
    """
    def ratio(a, b):
        return a / b if b else None
    return dict(iou=ratio(c["inter"], c["pred"] + c["ref"] - c["inter"]), dice=ratio(2 * c["inter"], c["pred"] + c["ref"]),
                surface_dice=ratio(c["surf_match"], c["surf"]))


def score(pred: np.ndarray, ref: np.ndarray, core: np.ndarray, body_proximity: np.ndarray,
          surface_dice: bool = True) -> dict:
    """Count one case's agreement with the reference, per region and class.

    Parameters
    ----------
    pred : numpy.ndarray
        Class index ``(nx, ny, nz)``: 0 none, 1 vortex, 2 shear layer.
    ref : numpy.ndarray
        The reference, in the same form.
    core : numpy.ndarray
        Bool ``(nx, ny, nz)``, the voxels that count.
    body_proximity : numpy.ndarray
        The body proximity channel, ``(nx, ny, nz)``.
    surface_dice : bool, optional
        False skips the surface counts, as in ``class_counts``.

    Returns
    -------
    dict
        ``{region: {class name: counts}}`` for the regions ``all`` (core voxels), ``near_body`` (within
        0.25 H of the body) and ``far`` (beyond 4 H from it).
    """
    regions = dict(all=core, near_body=core & (body_proximity >= NEAR_BODY), far=core & (body_proximity == 0))
    return {r: {name: class_counts(pred == k + 1, ref == k + 1, m, surface_dice) for k, name in enumerate(CLASS_NAMES)}
            for r, m in regions.items()}


def pool(scores: list[dict]) -> dict:
    """Sum the counts of several runs, which is how a split is scored.

    Parameters
    ----------
    scores : list[dict]
        Results of ``score``.

    Returns
    -------
    dict
        The same structure, with each count summed over the runs.
    """
    return {r: {c: {k: sum(s[r][c][k] for s in scores) for k in COUNTS} for c in CLASS_NAMES} for r in scores[0]}


def with_metrics(s: dict) -> dict:
    return {r: {c: {**counts, **metrics(counts)} for c, counts in classes.items()} for r, classes in s.items()}


def evaluate_run(
    path: Path, model: torch.nn.Module | None, contract: Contract, rules: LabelRules, device: torch.device,
    baseline: bool = False, probs: Path | None = None, surface_dice: bool = True,
) -> dict:
    """Score the net and, if asked, the baseline on one case.

    Parameters
    ----------
    path : pathlib.Path
        The case's Zarr store.
    model : torch.nn.Module or None
        The network in eval mode on ``device``. None skips the net.
    contract : Contract
        The input contract.
    rules : LabelRules
        The label thresholds, which define the core.
    device : torch.device
        Where the net runs.
    baseline : bool, optional
        Also score the voxel-grid baseline.
    probs : pathlib.Path or None, optional
        Write the net's vortex and shear-layer probabilities to a Zarr store here, as float32 arrays
        ``p_vortex`` and ``p_shear_layer`` on the case's grid. They are 0 outside the core.
    surface_dice : bool, optional
        False skips the surface counts, as in ``class_counts``.

    Returns
    -------
    dict
        ``net`` and ``baseline`` (when run), each as ``score`` returns.
    """
    store = zarr.open_group(path, mode="r")
    x, labels = store["inputs"][:], store["labels"][:]
    ref = labels[0] + 2 * labels[1]
    core = core_mask(x[4] < 1, (1 - x[5]) * contract.ground_proximity_range_H, rules, contract.n_per_H)[1]
    out = {}
    if model is not None:
        p = predict(model, torch.from_numpy(store["features"][:]), device) * torch.from_numpy(core).to(device)  # 0 outside the core
        out["net"] = score(p.argmax(0).cpu().numpy(), ref, core, x[4], surface_dice)
        if probs is not None:
            out_store = zarr.open_group(probs, mode="w")
            grid = store.attrs["grid"]
            # origin and spacing in x, y, z order and dimension_names, so the ParaView Zarr reader places
            # the arrays on the case's grid: without names it reads a 3D array as (z, y, x).
            out_store.attrs.update(origin=grid["origin"], spacing=grid["dx"])
            for name, a in zip(("p_vortex", "p_shear_layer"), p[1:].cpu().numpy()):
                out_store.create_array(name, data=a, dimension_names=("x", "y", "z"))
    if baseline:
        out["baseline"] = score(baseline_classes(x, core, contract, rules), ref, core, x[4], surface_dice)
    return out


def file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Run the ``cfd-rgba-net-evaluate`` command line.

    Parameters
    ----------
    argv : list[str] or None, optional
        The arguments. ``None`` means ``sys.argv[1:]``.

    Returns
    -------
    int
        The exit code, 0 on success.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="model weights; without them only the baseline is scored")
    ap.add_argument("--split", required=True, help="a value of the split CSV's 'split' column")
    ap.add_argument("--csv", type=Path, default=configs_dir() / "splits" / "ahmedml_v1.csv")
    ap.add_argument("--stores", type=Path, required=True, help="the folder of case stores")
    ap.add_argument("--out", type=Path, required=True, help="where the JSON files go")
    ap.add_argument("--dataset", default="ahmedml", help="dataset config: a name in configs/datasets/ or a YAML path")
    ap.add_argument("--runs", help="score only these runs of the split, as in cfd-rgba-net-batch --runs")
    ap.add_argument("--baseline", action="store_true", help="also score the voxel-grid baseline")
    ap.add_argument("--probs", action="store_true", help="also write each run's probabilities to --out")
    # the checkpoint's architecture, as the pilot's flags of the same names set it
    ap.add_argument("--levels", type=int, default=5, help="resolution levels of the UNet")
    ap.add_argument("--prior", action="store_true", help="the net adds the rules' margins to its logits")
    args = ap.parse_args(argv)
    if not (args.checkpoint or args.baseline):
        ap.error("give --checkpoint, --baseline or both")

    df = pd.read_csv(args.csv)
    runs = df.run[(df.split == args.split) & df.scored].tolist()
    if args.runs:
        runs = [r for r in parse_runs(args.runs) if r in runs]
    if not runs:
        ap.error(f"no scored runs in split {args.split!r}" + (" among --runs" if args.runs else ""))
    cfg, contract, rules = CaseConfig.load(args.dataset), Contract.load(), LabelRules.load()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if args.checkpoint:
        # the prior's scales are learned, so they come from the checkpoint, not from a flag
        model = UNet3D(rules.tau_lambda2, rules.tau_omega, args.levels, prior=args.prior).to(device).eval()
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    meta = dict(**git_state(), split=args.split, csv_sha256=file_sha256(args.csv),
                checkpoint=str(args.checkpoint) if args.checkpoint else None,
                checkpoint_sha256=file_sha256(args.checkpoint) if args.checkpoint else None,
                model=dict(levels=args.levels, prior=args.prior) if args.checkpoint else None)

    args.out.mkdir(parents=True, exist_ok=True)
    scores = []
    for run in runs:
        probs = args.out / f"run{run}_probs.zarr" if args.probs else None
        scores.append(evaluate_run(store_path(cfg, args.stores, run), model, contract, rules, device,
                                   args.baseline, probs))
        (args.out / f"run{run}.json").write_text(
            json.dumps(dict(run=run, **meta, **{k: with_metrics(v) for k, v in scores[-1].items()}), indent=1))
        print(f"run {run} done", flush=True)
    pooled = {k: with_metrics(pool([s[k] for s in scores])) for k in scores[0]}
    (args.out / "split.json").write_text(json.dumps(dict(**meta, runs=runs, **pooled), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
