"""
Figures for progress.md section III (data trends and distributions), from the AhmedML case stores
(<out>/ahmedml_run<N>.zarr, written by cfd-rgba-net-batch).

    slant      box plots of per-case class fraction vs. slant-angle bin   (store attributes only)
    sections   y=0 and crossflow slices of u/Uinf and labels for two runs (2-D slices of two stores)
    hist       per-class histograms of the five inputs                    (voxel sample)
    corr       Spearman correlation of inputs, ingredients and labels     (voxel sample)

The voxel sample is --per-run random fluid voxels (seeded by run number) from each store: inputs
(u, v, w, Cp, sdf), ingredients (Q*, lambda2*, |omega|*) and labels (vortex, shear_layer). It is
drawn once per invocation and shared by hist and corr.

Usage:
    python scripts/ahmedml/figures.py slant sections hist corr \\
        --out /data/cfd-rgba-net-data/ahmedML/out_v1 --fig figures
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL = {"vortex": "vortex", "shear_layer": "shear layer"}
COLOR = {"vortex": "#eb6834", "shear_layer": "#1baf7a"}
FREE = "#52514e"
BODY = "#3a3a38"
DIVERGING = LinearSegmentedColormap.from_list("bluered", ["#d03b3b", "#f0efec", "#2a78d6"])
INGREDIENTS = ["Q_nd", "lambda2_nd", "omega_mag_nd"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "figure.titlesize": 13, "legend.frameon": False,
})


def stores(out):
    """run -> path of every complete store in out (an unfinished one is still named *.zarr.tmp)."""
    return {int(p.stem.rsplit("_run", 1)[1]): p for p in sorted(Path(out).glob("ahmedml_run*.zarr"))}


def open_store(path):
    return zarr.open_group(path, mode="r")


def load_attrs(out):
    def one(p):
        return dict(open_store(p).attrs)
    paths = stores(out)
    with ThreadPoolExecutor(16) as ex:
        return dict(zip(paths, ex.map(one, paths.values())))


def slant_of(a):
    return a["geometry"]["geo_parameters"]["slant-angle-degrees"]


def fig_slant(out, fig_dir):
    attrs = load_attrs(out)
    classes = next(iter(attrs.values()))["class_names"]
    slant = np.array([slant_of(a) for a in attrs.values()])
    frac = {c: np.array([100 * a["class_fractions_fluid"][c] for a in attrs.values()]) for c in classes}
    edges = np.arange(5, 75, 5)
    centers = (edges[:-1] + edges[1:]) / 2
    which = np.digitize(np.clip(slant, edges[0], edges[-1] - 1e-6), edges) - 1  # 70.4° folds into the last bin
    fig, axes = plt.subplots(len(classes), 1, figsize=(9, 2.25 * len(classes) + 0.5), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, c in zip(axes, classes):
        data = [frac[c][which == b] for b in range(len(centers))]
        bp = ax.boxplot(data, positions=centers, widths=3.2, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=COLOR[c], markeredgecolor="none", alpha=0.6))
        for b in bp["boxes"]:
            b.set(facecolor=COLOR[c], edgecolor=SURF, linewidth=1.5, alpha=0.85)
        ax.set_ylabel(f"{LABEL[c]}\n[% fluid voxels]")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    counts = [int((which == b).sum()) for b in range(len(centers))]
    for x, n in zip(centers, counts):
        axes[0].text(x, 1.02, f"n={n}", transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=INK2)
    for ax in axes:
        ax.set_xticks(edges, [str(e) for e in edges])
        ax.set_xlim(edges[0], edges[-1])
    axes[-1].set_xlabel("slant angle [deg] (5° bins)")
    fig.suptitle(f"Share of fluid voxels per label class vs. slant angle ({len(attrs)} runs)", y=0.995)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/class_fraction_vs_slant.png", dpi=150)
    plt.close(fig)


def label_index(lab, classes):
    """Collapse the multi-label (C, ...) array to one index per voxel: 0 none, then classes in order, earlier wins."""
    idx = np.zeros(lab.shape[1:], np.uint8)
    for i in reversed(range(len(classes))):
        idx[lab[i] > 0] = i + 1
    return idx


def fig_sections(out, fig_dir, runs):
    paths = stores(out)
    missing = [r for r in runs if r not in paths]
    if missing:
        raise SystemExit(f"no store for runs {missing} in {out}")
    classes = None
    fig, axes = plt.subplots(len(runs), 4, figsize=(17, 2.6 * len(runs) + 1.4), layout="constrained",
                             gridspec_kw=dict(width_ratios=[2.6, 2.6, 1, 1]), squeeze=False)
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.5)
    for row, run in zip(axes, runs):
        g = open_store(paths[run])
        a = dict(g.attrs)
        classes = a["class_names"]
        lab_cmap = ListedColormap([SURF] + [COLOR[c] for c in classes] + [BODY])
        n_idx = len(classes) + 1  # index of "body" in lab_cmap
        o, dx = np.array(a["grid"]["origin"]), a["grid"]["dx"]
        nx, ny, nz = a["grid"]["dims"]
        H = a["geometry"]["H"]
        x_nose, x_base = a["geometry"]["body_bounds"][0] / H, a["geometry"]["body_bounds"][1] / H
        xs = (o[0] + np.arange(nx) * dx) / H
        ys = (o[1] + np.arange(ny) * dx) / H
        zs = (o[2] + np.arange(nz) * dx) / H
        iz = zs <= 2.0
        u_ch = a["channel_names"].index("u")

        def section(sel):
            """u/Uinf (masked in the body) and label index (n_idx in the body) on one plane."""
            body = ~g["fluid"][sel]
            u = np.ma.masked_where(body, g["inputs"][(u_ch,) + sel])
            lab = label_index(g["labels"][(slice(None),) + sel], classes)
            lab[body] = n_idx
            return u, lab

        # symmetry plane y = 0, cropped to nose - 0.5H .. base + 4H, z <= 2H
        j = int(np.argmin(np.abs(ys)))
        u, lab = section((slice(None), j, slice(None)))
        ix = (xs >= x_nose - 0.5) & (xs <= x_base + 4)
        ext = [xs[ix][0], xs[ix][-1], zs[iz][0], zs[iz][-1]]
        im = row[0].imshow(u[ix][:, iz].T, origin="lower", extent=ext, cmap=DIVERGING, norm=norm, aspect="equal", interpolation="nearest")
        row[0].set_facecolor(BODY)
        row[1].imshow(lab[ix][:, iz].T, origin="lower", extent=ext, cmap=lab_cmap, vmin=-0.5, vmax=n_idx + 0.5, aspect="equal", interpolation="nearest")
        row[0].set_title(f"run {run}, slant {slant_of(a):.1f}°: u/U∞ on y = 0")
        row[1].set_title(f"run {run}: labels on y = 0")
        # crossflow plane 1 H behind the base, |y| <= 1.5 H
        i = int(np.argmin(np.abs(xs - (x_base + 1.0))))
        u, lab = section((i, slice(None), slice(None)))
        iy = np.abs(ys) <= 1.5
        ext2 = [ys[iy][0], ys[iy][-1], zs[iz][0], zs[iz][-1]]
        row[2].imshow(u[iy][:, iz].T, origin="lower", extent=ext2, cmap=DIVERGING, norm=norm, aspect="equal", interpolation="nearest")
        row[2].set_facecolor(BODY)
        row[3].imshow(lab[iy][:, iz].T, origin="lower", extent=ext2, cmap=lab_cmap, vmin=-0.5, vmax=n_idx + 0.5, aspect="equal", interpolation="nearest")
        row[2].set_title("u/U∞ at x = base + 1 H")
        row[3].set_title("labels at x = base + 1 H")
        for ax in row[:2]:
            ax.set_xlabel("x / H"); ax.set_ylabel("z / H")
        for ax in row[2:]:
            ax.set_xlabel("y / H"); ax.set_ylabel("z / H")
    cb = fig.colorbar(im, ax=axes[-1, 0], location="bottom", shrink=0.9, aspect=40)
    cb.set_label("u / U∞  (red = reversed flow)")
    handles = [Patch(color=COLOR[c], label=LABEL[c]) for c in classes] + [Patch(color=BODY, label="body")]
    fig.legend(handles=handles, loc="outside lower right", ncol=len(handles))
    title = "Symmetry-plane (y = 0) and crossflow sections"
    if len(runs) == 2:
        title += f": attached-vortex regime (run {runs[0]}) vs. separated regime (run {runs[1]})"
    fig.suptitle(title)
    fig.savefig(f"{fig_dir}/sections_run{'_'.join(map(str, runs))}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sample_run(run, path, n):
    """n random fluid voxels of one store: (inputs (5, n), ingredients (3, n), labels (C, n) bool, channel and class names)."""
    g = open_store(path)
    a = dict(g.attrs)
    fluid = np.flatnonzero(g["fluid"][...])
    rng = np.random.default_rng(run)
    k = np.sort(rng.choice(fluid, min(n, fluid.size), replace=False))
    ijk = np.unravel_index(k, g["fluid"].shape)

    def take(arr):
        """arr[:, ijk] as (C, n) in one coordinate read, so each chunk is decompressed once."""
        c = arr.shape[0]
        coords = (np.repeat(np.arange(c), k.size),) + tuple(np.tile(q, c) for q in ijk)
        return arr.vindex[coords].reshape(c, k.size)

    x = take(g["inputs"])
    ing = np.stack([g["ingredients"][name].vindex[ijk] for name in INGREDIENTS])
    y = take(g["labels"]).astype(bool)
    return x, ing, y, a["channel_names"], a["class_names"]


def load_sample(out, per_run, max_runs=None, workers=8):
    paths = stores(out)
    runs = sorted(paths)[:max_runs]
    print(f"sampling {per_run} fluid voxels from each of {len(runs)} stores", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda r: sample_run(r, paths[r], per_run), runs))
    x = np.concatenate([p[0] for p in parts], 1)
    ing = np.concatenate([p[1] for p in parts], 1)
    y = np.concatenate([p[2] for p in parts], 1)
    return dict(x=x, ing=ing, y=y, channels=parts[0][3], classes=parts[0][4], ncase=len(runs))


NAMES = {"u": "u / U∞", "v": "v / U∞", "w": "w / U∞", "Cp": "C_p", "sdf": "sdf / H"}
LIMS = {"u": (-0.6, 1.5), "v": (-0.6, 0.6), "w": (-0.6, 0.6), "Cp": (-1.2, 1.05), "sdf": (0.0, 4.0)}


def fig_hist(s, fig_dir):
    x, y, channels, classes = s["x"], s["y"], s["channels"], s["classes"]
    none = ~y.any(0)
    ncol = 3
    nrow = -(-len(channels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4 * nrow), layout="constrained", squeeze=False)
    for ax, f, ch in zip(axes.flat, range(len(channels)), channels):
        lo, hi = LIMS[ch]
        bins = np.linspace(lo, hi, 121)
        ax.hist(np.clip(x[f, none], lo, hi), bins=bins, density=True, histtype="step", color=FREE, linewidth=2, linestyle="--", label="no label")
        for i, c in enumerate(classes):
            ax.hist(np.clip(x[f, y[i]], lo, hi), bins=bins, density=True, histtype="step", color=COLOR[c], linewidth=2, label=LABEL[c])
        ax.set_yscale("log")
        ax.set_xlabel(f"{NAMES[ch]} (clipped to [{lo}, {hi}])")
        ax.set_ylabel("probability density (log)")
        ax.set_title(f"{NAMES[ch]} by label class")
        ax.grid(axis="y")
    for ax in axes.flat[len(channels):]:
        ax.set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(h))
    fig.suptitle(f"Input feature distributions per class ({x.shape[1]/1e6:.1f}M random fluid voxels from {s['ncase']} runs)")
    fig.savefig(f"{fig_dir}/feature_hist_by_class.png", dpi=150)
    plt.close(fig)


def fig_corr(s, fig_dir):
    from scipy.stats import spearmanr
    x, ing, y, channels, classes = s["x"], s["ing"], s["y"], s["channels"], s["classes"]
    rng = np.random.default_rng(0)
    k = rng.choice(x.shape[1], min(2_000_000, x.shape[1]), replace=False)
    cols = [x[i, k] for i in range(len(channels))] + [ing[i, k] for i in range(len(INGREDIENTS))] \
        + [y[i, k].astype(np.float32) for i in range(len(classes))]
    names = [NAMES[c].replace(" ", "") for c in channels] + ["Q*", "λ₂*", "|ω|*"] + [LABEL[c] for c in classes]
    r = spearmanr(np.stack(cols, 1)).statistic
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(r, cmap=DIVERGING.reversed(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.spines[:].set_visible(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK if abs(r[i, j]) < 0.6 else SURF)
    for p in [len(channels) - 0.5, len(channels) + len(INGREDIENTS) - 0.5]:
        ax.axhline(p, color=SURF, linewidth=3); ax.axvline(p, color=SURF, linewidth=3)
    ax.set_xlabel("inputs  |  label ingredients  |  labels")
    ax.set_ylabel("labels  |  label ingredients  |  inputs")  # rotated label reads bottom-to-top
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Spearman rank correlation")
    ax.set_title(f"Spearman correlation ({len(k)/1e6:.1f}M random fluid voxels, {s['ncase']} runs)")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/correlation_heatmap.png", dpi=150)
    plt.close(fig)
    np.savetxt(f"{fig_dir}/correlation_heatmap.csv", r, delimiter=",", header=",".join(names), fmt="%.4f")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figs", nargs="+", choices=["slant", "sections", "hist", "corr"])
    ap.add_argument("--out", required=True, help="folder of ahmedml_run<N>.zarr stores")
    ap.add_argument("--fig", required=True, help="where the figures are written")
    ap.add_argument("--runs", type=int, nargs="+", default=[420, 346], help="runs shown by sections")
    ap.add_argument("--per-run", type=int, default=20_000, help="fluid voxels sampled per store for hist and corr")
    ap.add_argument("--max-runs", type=int, default=None, help="sample only the first this many stores")
    ap.add_argument("--workers", type=int, default=8, help="stores sampled at once")
    a = ap.parse_args()
    Path(a.fig).mkdir(parents=True, exist_ok=True)
    sample = None
    for f in a.figs:
        if f == "slant":
            fig_slant(a.out, a.fig)
        elif f == "sections":
            fig_sections(a.out, a.fig, a.runs)
        else:
            if sample is None:
                sample = load_sample(a.out, a.per_run, a.max_runs, a.workers)
            (fig_hist if f == "hist" else fig_corr)(sample, a.fig)
        print(f"wrote {f}", flush=True)


if __name__ == "__main__":
    main()
