"""
Figures for progress.md section III (data trends and distributions).

    slant      box plots of per-case class fraction vs. slant-angle bin   (reads *_meta.json)
    sections   y=0 and crossflow slices of u/Uinf and labels for two runs (reads *_inputs/_labels.npz)
    hist       per-class histograms of the four inputs                    (reads a voxel sample dir)
    corr       Spearman correlation of inputs, ingredients and labels     (reads a voxel sample dir)

The voxel sample is a directory of s_<run>.npz files, each holding 20k random fluid voxels
with x (u, v, w, Cp), ing (Q*, lambda2*, |omega|*, Cpt) and y (4 labels).

Usage:
    python progress_figures.py slant sections --out /data/cfd-rgba-net-data/ahmedML/out --fig figures
    python progress_figures.py hist corr --sample <dir> --fig figures
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

# Palette: 3 categorical slots (validated all-pairs) + neutral gray for wake, which overlaps the others.
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CLASSES = ["recirculation", "vortex", "shear_layer", "wake"]
LABEL = {"recirculation": "recirculation", "vortex": "vortex", "shear_layer": "shear layer", "wake": "wake"}
COLOR = {"recirculation": "#2a78d6", "vortex": "#eb6834", "shear_layer": "#1baf7a", "wake": "#9a9893"}
FREE = "#52514e"
BODY = "#3a3a38"
DIVERGING = LinearSegmentedColormap.from_list("bluered", ["#d03b3b", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "figure.titlesize": 13, "legend.frameon": False,
})


def load_meta(out):
    return {m["run"]: m for m in (json.loads(Path(f).read_text()) for f in glob.glob(f"{out}/ahmed_run*_meta.json"))}


def fig_slant(out, fig_dir):
    metas = load_meta(out)
    slant = np.array([m["geo_parameters"]["slant-angle-degrees"] for m in metas.values()])
    frac = {c: np.array([100 * m["class_fractions_fluid"][c] for m in metas.values()]) for c in CLASSES}
    edges = np.arange(5, 75, 5)
    centers = (edges[:-1] + edges[1:]) / 2
    which = np.digitize(np.clip(slant, edges[0], edges[-1] - 1e-6), edges) - 1  # 70.4° folds into the last bin
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for ax, c in zip(axes, CLASSES):
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
    fig.suptitle("Share of fluid voxels per label class vs. slant angle (499 runs)", y=0.995)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/class_fraction_vs_slant.png", dpi=150)
    plt.close(fig)


def label_index(lab):
    """Collapse multi-label (4, ...) to one index per voxel for display. Priority: vortex > shear > recirc > wake."""
    idx = np.zeros(lab.shape[1:], np.uint8)  # 0 none, 1 vortex, 2 shear, 3 recirc, 4 wake; later writes win
    idx[lab[3] > 0] = 4
    idx[lab[0] > 0] = 3
    idx[lab[2] > 0] = 2
    idx[lab[1] > 0] = 1
    return idx


LAB_CMAP = ListedColormap([SURF, COLOR["vortex"], COLOR["shear_layer"], COLOR["recirculation"], "#c9c8c3", BODY])


def fig_sections(out, fig_dir, runs):
    metas = load_meta(out)
    fig, axes = plt.subplots(len(runs), 4, figsize=(17, 2.6 * len(runs) + 1.4), layout="constrained",
                             gridspec_kw=dict(width_ratios=[2.6, 2.6, 1, 1]))
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.5)
    for row, run in zip(axes, runs):
        m = metas[run]
        tag = m["tag"]
        X = np.load(f"{out}/{tag}_inputs.npz")["inputs"]
        Y = np.load(f"{out}/{tag}_labels.npz")["labels"]
        o, dx, H, L = np.array(m["origin"]), m["dx"], m["H"], m["L"]
        nx, ny, nz = m["dims"]
        xs = (o[0] + np.arange(nx) * dx) / H
        ys = (o[1] + np.arange(ny) * dx) / H
        zs = (o[2] + np.arange(nz) * dx) / H
        body = X[4] < 0.5
        lab = label_index(Y)
        lab[body] = 5
        u = np.ma.masked_where(body, X[0])
        # symmetry plane y = 0, cropped to nose - 0.5H .. base + 4H, z <= 2H
        j = int(np.argmin(np.abs(ys)))
        xb = m["x_base"] / H
        ix = (xs >= -L / H - 0.5) & (xs <= xb + 4)
        iz = zs <= 2.0
        ext = [xs[ix][0], xs[ix][-1], zs[iz][0], zs[iz][-1]]
        im = row[0].imshow(u[ix][:, j][:, iz].T, origin="lower", extent=ext, cmap=DIVERGING, norm=norm, aspect="equal", interpolation="nearest")
        row[0].set_facecolor(BODY)
        row[1].imshow(lab[ix][:, j][:, iz].T, origin="lower", extent=ext, cmap=LAB_CMAP, vmin=-0.5, vmax=5.5, aspect="equal", interpolation="nearest")
        slant = m["geo_parameters"]["slant-angle-degrees"]
        row[0].set_title(f"run {run}, slant {slant:.1f}°: u/U∞ on y = 0")
        row[1].set_title(f"run {run}: labels on y = 0")
        # crossflow plane 1 H behind the base, |y| <= 1.5 H
        i = int(np.argmin(np.abs(xs - (xb + 1.0))))
        iy = np.abs(ys) <= 1.5
        ext2 = [ys[iy][0], ys[iy][-1], zs[iz][0], zs[iz][-1]]
        row[2].imshow(u[i][iy][:, iz].T, origin="lower", extent=ext2, cmap=DIVERGING, norm=norm, aspect="equal", interpolation="nearest")
        row[3].imshow(lab[i][iy][:, iz].T, origin="lower", extent=ext2, cmap=LAB_CMAP, vmin=-0.5, vmax=5.5, aspect="equal", interpolation="nearest")
        row[2].set_title(f"u/U∞ at x = base + 1 H")
        row[3].set_title(f"labels at x = base + 1 H")
        for a in row[:2]:
            a.set_xlabel("x / H"); a.set_ylabel("z / H")
        for a in row[2:]:
            a.set_xlabel("y / H"); a.set_ylabel("z / H")
    cb = fig.colorbar(im, ax=axes[-1, 0], location="bottom", shrink=0.9, aspect=40)
    cb.set_label("u / U∞  (red = reversed flow)")
    handles = [Patch(color=COLOR[c], label=LABEL[c]) for c in ["recirculation", "vortex", "shear_layer"]]
    handles += [Patch(color="#c9c8c3", label="wake (only)"), Patch(color=BODY, label="body")]
    fig.legend(handles=handles, loc="outside lower right", ncol=5)
    fig.suptitle(f"Symmetry-plane (y = 0) and crossflow sections: attached-vortex regime (run {runs[0]}) vs. separated regime (run {runs[1]})")
    fig.savefig(f"{fig_dir}/sections_run{'_'.join(map(str, runs))}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_sample(sample):
    fs = sorted(glob.glob(f"{sample}/s_*.npz"))
    d = [np.load(f) for f in fs]
    x = np.concatenate([a["x"] for a in d], 1)
    ing = np.concatenate([a["ing"] for a in d], 1)
    y = np.concatenate([a["y"] for a in d], 1).astype(bool)
    return x, ing, y, len(fs)


def fig_hist(sample, fig_dir):
    x, _, y, ncase = load_sample(sample)
    names = ["u / U∞", "v / U∞", "w / U∞", "C_p"]
    lims = [(-0.6, 1.5), (-0.6, 0.6), (-0.6, 0.6), (-1.2, 1.05)]
    none = ~y.any(0)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, f, name, (lo, hi) in zip(axes.flat, range(4), names, lims):
        bins = np.linspace(lo, hi, 121)
        ax.hist(np.clip(x[f, none], lo, hi), bins=bins, density=True, histtype="step", color=FREE, linewidth=2, linestyle="--", label="free stream (no label)")
        for i, c in enumerate(CLASSES):
            ax.hist(np.clip(x[f, y[i]], lo, hi), bins=bins, density=True, histtype="step", color=COLOR[c], linewidth=2, label=LABEL[c])
        ax.set_yscale("log")
        ax.set_xlabel(f"{name} (clipped to [{lo}, {hi}])")
        ax.set_ylabel("probability density (log)")
        ax.set_title(f"{name} by label class")
        ax.grid(axis="y")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5)
    fig.suptitle(f"Input feature distributions per class ({x.shape[1]/1e6:.1f}M random fluid voxels from {ncase} runs)")
    fig.savefig(f"{fig_dir}/feature_hist_by_class.png", dpi=150)
    plt.close(fig)


def fig_corr(sample, fig_dir):
    from scipy.stats import spearmanr
    x, ing, y, ncase = load_sample(sample)
    rng = np.random.default_rng(0)
    k = rng.choice(x.shape[1], min(2_000_000, x.shape[1]), replace=False)
    cols = [x[0, k], x[1, k], x[2, k], x[3, k], ing[0, k], ing[1, k], ing[2, k], ing[3, k]] + [y[i, k].astype(np.float32) for i in range(4)]
    names = ["u/U∞", "v/U∞", "w/U∞", "C_p", "Q*", "λ₂*", "|ω|*", "C_pt"] + [LABEL[c] for c in CLASSES]
    r = spearmanr(np.stack(cols, 1)).statistic
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(r, cmap=DIVERGING.reversed(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.spines[:].set_visible(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK if abs(r[i, j]) < 0.6 else SURF)
    for p in [3.5, 7.5]:
        ax.axhline(p, color=SURF, linewidth=3); ax.axvline(p, color=SURF, linewidth=3)
    ax.set_xlabel("inputs  |  label ingredients  |  labels")
    ax.set_ylabel("labels  |  label ingredients  |  inputs")  # rotated label reads bottom-to-top
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Spearman rank correlation")
    ax.set_title(f"Spearman correlation ({len(k)/1e6:.0f}M random fluid voxels, {ncase} runs)")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/correlation_heatmap.png", dpi=150)
    plt.close(fig)
    np.savetxt(f"{fig_dir}/correlation_heatmap.csv", r, delimiter=",", header=",".join(names), fmt="%.4f")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figs", nargs="+", choices=["slant", "sections", "hist", "corr"])
    ap.add_argument("--out", default="/data/cfd-rgba-net-data/ahmedML/out")
    ap.add_argument("--sample", default=None)
    ap.add_argument("--fig", default="figures")
    ap.add_argument("--runs", type=int, nargs="+", default=[420, 346])
    a = ap.parse_args()
    Path(a.fig).mkdir(parents=True, exist_ok=True)
    for f in a.figs:
        if f == "slant": fig_slant(a.out, a.fig)
        if f == "sections": fig_sections(a.out, a.fig, a.runs)
        if f == "hist": fig_hist(a.sample, a.fig)
        if f == "corr": fig_corr(a.sample, a.fig)


if __name__ == "__main__":
    main()
