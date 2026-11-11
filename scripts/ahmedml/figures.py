"""
Progress figures (data trends and distributions) from the AhmedML case stores
(<out>/ahmedml_run<N>.zarr, written by cfd-rgba-net-batch).

    slant      box plots of per-case class fraction vs. slant-angle bin   (store attributes only)
    sections   y=0 and crossflow slices of u/Uinf and labels for two runs (2-D slices of two stores)
    hist       per-class histograms of the input channels                 (voxel sample)
    corr       Spearman correlation of inputs, ingredients and labels     (voxel sample)

The voxel sample is --per-run random voxels outside the body (body_proximity < 1; seeded by run number)
from each store: inputs (u, v, w, Cp, body_proximity, ground_proximity), ingredients (Q*, lambda2*,
|omega|*) and labels (vortex, shear_layer). It is drawn once per invocation and shared by hist and corr.

Needs matplotlib, which is not a declared dependency of the project: it is installed with vtk.

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
INGREDIENT_NAMES = {"Q_nd": "Q*", "lambda2_nd": "λ₂*", "omega_mag_nd": "|ω|*"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "figure.titlesize": 13, "legend.frameon": False,
})


def stores(out):
    """Find the finished stores of a folder.

    An unfinished store is still named ``*.zarr.tmp``, so it does not match.

    Parameters
    ----------
    out : str or pathlib.Path
        The folder of ``ahmedml_run<N>.zarr`` stores.

    Returns
    -------
    dict[int, pathlib.Path]
        The path of each store by run number, in path order.

    Raises
    ------
    SystemExit
        If the folder holds no store.
    """
    found = {int(p.stem.rsplit("_run", 1)[1]): p for p in sorted(Path(out).glob("ahmedml_run*.zarr"))}
    if not found:
        raise SystemExit(f"no ahmedml_run<N>.zarr stores in {out}")
    return found


def open_store(path):
    """Open a store read-only.

    Parameters
    ----------
    path : str or pathlib.Path
        The store.

    Returns
    -------
    zarr.Group
        The root group.
    """
    return zarr.open_group(path, mode="r")


def load_attrs(out):
    """Read the attributes of every store, 16 stores at a time.

    Parameters
    ----------
    out : str or pathlib.Path
        The folder of stores.

    Returns
    -------
    dict[int, dict]
        The root attributes of each store by run number.
    """
    def one(p):
        return dict(open_store(p).attrs)
    paths = stores(out)
    with ThreadPoolExecutor(16) as ex:
        return dict(zip(paths, ex.map(one, paths.values())))


def slant_of(a):
    """Read a case's slant angle.

    Parameters
    ----------
    a : dict
        The store attributes.

    Returns
    -------
    float
        The slant angle, in degrees.

    Raises
    ------
    KeyError
        If the store holds no geometry parameters, for example because their CSV was missing at build time.
    """
    return a["geometry"]["geo_parameters"]["slant-angle-degrees"]


def fig_slant(out, fig_dir):
    """Plot each class's share of the fluid voxels against the slant angle.

    One box plot per 5-degree bin, from 5 to 70 degrees; a slant outside that range falls into the first
    or last bin. Writes ``class_fraction_vs_slant.png``.

    Parameters
    ----------
    out : str or pathlib.Path
        The folder of stores.
    fig_dir : str or pathlib.Path
        The folder the figure is written to.
    """
    attrs = load_attrs(out)
    classes = next(iter(attrs.values()))["class_names"]
    slant = np.array([slant_of(a) for a in attrs.values()])
    frac = {c: np.array([100 * a["class_fractions_fluid"][c] for a in attrs.values()]) for c in classes}
    edges = np.arange(5, 75, 5)
    centers = (edges[:-1] + edges[1:]) / 2
    # slants outside the edges fold into the end bins (the dataset has 70.4 degrees)
    which = np.digitize(np.clip(slant, edges[0], edges[-1] - 1e-6), edges) - 1
    fig, axes = plt.subplots(len(classes), 1, figsize=(9, 2.25 * len(classes) + 0.5), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, c in zip(axes, classes):
        data = [frac[c][which == b] for b in range(len(centers))]
        bp = ax.boxplot(data, positions=centers, widths=3.2, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=COLOR[c],
                                        markeredgecolor="none", alpha=0.6))
        for b in bp["boxes"]:
            b.set(facecolor=COLOR[c], edgecolor=SURF, linewidth=1.5, alpha=0.85)
        ax.set_ylabel(f"{LABEL[c]}\n[% fluid voxels]")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    counts = [int((which == b).sum()) for b in range(len(centers))]
    for x, n in zip(centers, counts):
        axes[0].text(x, 1.02, f"n={n}", transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=8, color=INK2)
    for ax in axes:
        ax.set_xticks(edges, [str(e) for e in edges])
        ax.set_xlim(edges[0], edges[-1])
    axes[-1].set_xlabel("slant angle [deg] (5° bins)")
    fig.suptitle(f"Share of fluid voxels per label class vs. slant angle ({len(attrs)} runs)", y=0.995)
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "class_fraction_vs_slant.png", dpi=150)
    plt.close(fig)


def label_index(lab, classes):
    """Collapse the multi-label array to one class index per voxel.

    Parameters
    ----------
    lab : numpy.ndarray
        Labels, shape ``(C, ...)``, non-zero where class ``i`` applies.
    classes : sequence of str
        The class names, in the order of the first axis of ``lab``.

    Returns
    -------
    numpy.ndarray
        Shape ``lab.shape[1:]``, dtype uint8: 0 for no class, ``i + 1`` for class ``i``. Where several
        classes apply, the earlier one wins.
    """
    idx = np.zeros(lab.shape[1:], np.uint8)
    for i in reversed(range(len(classes))):
        idx[lab[i] > 0] = i + 1
    return idx


def fig_sections(out, fig_dir, runs):
    """Plot two slices of u/U∞ and of the labels for each of some runs.

    Each run is a row of four panels: the symmetry plane y = 0, cropped to the body's nose minus 0.5 H
    through its base plus 4 H and to z <= 2 H, and the crossflow plane 1 H behind the base with
    |y| <= 1.5 H. The body is masked in the velocity panels and drawn dark in the label panels. Assumes
    the case is oriented as AhmedML is: flow along +x, up along +z. Writes ``sections_run<runs>.png``.

    Parameters
    ----------
    out : str or pathlib.Path
        The folder of stores.
    fig_dir : str or pathlib.Path
        The folder the figure is written to.
    runs : list[int]
        The runs to show, one row each.

    Raises
    ------
    SystemExit
        If a run has no store.
    """
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
        b_ch = a["channel_names"].index("body_proximity")

        def section(sel):
            """u/Uinf (masked in the body) and label index (n_idx in the body) on one plane."""
            body = g["inputs"][(b_ch,) + sel] > 1  # inside the body: body_proximity > 1
            u = np.ma.masked_where(body, g["inputs"][(u_ch,) + sel])
            lab = label_index(g["labels"][(slice(None),) + sel], classes)
            lab[body] = n_idx
            return u, lab

        # symmetry plane y = 0, cropped to nose - 0.5H .. base + 4H, z <= 2H
        j = int(np.argmin(np.abs(ys)))
        u, lab = section((slice(None), j, slice(None)))
        ix = (xs >= x_nose - 0.5) & (xs <= x_base + 4)
        ext = [xs[ix][0], xs[ix][-1], zs[iz][0], zs[iz][-1]]
        im = row[0].imshow(u[ix][:, iz].T, origin="lower", extent=ext, cmap=DIVERGING, norm=norm,
                           aspect="equal", interpolation="nearest")
        row[0].set_facecolor(BODY)
        row[1].imshow(lab[ix][:, iz].T, origin="lower", extent=ext, cmap=lab_cmap, vmin=-0.5, vmax=n_idx + 0.5,
                      aspect="equal", interpolation="nearest")
        row[0].set_title(f"run {run}, slant {slant_of(a):.1f}°: u/U∞ on y = 0")
        row[1].set_title(f"run {run}: labels on y = 0")
        # crossflow plane 1 H behind the base, |y| <= 1.5 H
        i = int(np.argmin(np.abs(xs - (x_base + 1.0))))
        u, lab = section((i, slice(None), slice(None)))
        iy = np.abs(ys) <= 1.5
        ext2 = [ys[iy][0], ys[iy][-1], zs[iz][0], zs[iz][-1]]
        row[2].imshow(u[iy][:, iz].T, origin="lower", extent=ext2, cmap=DIVERGING, norm=norm,
                      aspect="equal", interpolation="nearest")
        row[2].set_facecolor(BODY)
        row[3].imshow(lab[iy][:, iz].T, origin="lower", extent=ext2, cmap=lab_cmap, vmin=-0.5, vmax=n_idx + 0.5,
                      aspect="equal", interpolation="nearest")
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
    fig.suptitle("Symmetry-plane (y = 0) and crossflow sections")
    fig.savefig(Path(fig_dir) / f"sections_run{'_'.join(map(str, runs))}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sample_run(run, path, n):
    """Draw random voxels outside the body from one store.

    "Outside the body" is ``body_proximity < 1``. It also includes any voxel outside the mesh, whose
    inputs are zero. AhmedML's domain is the whole mesh, so there are almost none. The draw is seeded by
    the run number, so it is reproducible. Voxels that no class applies to include those the label rules
    exclude near walls and the floor.

    Parameters
    ----------
    run : int
        The run number, which seeds the draw.
    path : str or pathlib.Path
        The store.
    n : int
        Number of voxels to draw. Fewer if the store has fewer.

    Returns
    -------
    x : numpy.ndarray
        Input channels, shape ``(C_in, n)``.
    ing : numpy.ndarray
        Label ingredients ``Q*``, ``lambda2*`` and ``|omega|*``, shape ``(3, n)``.
    y : numpy.ndarray
        Boolean labels, shape ``(C, n)``.
    channels : list[str]
        The input channel names.
    classes : list[str]
        The class names.
    """
    g = open_store(path)
    a = dict(g.attrs)
    b = g["inputs"][a["channel_names"].index("body_proximity")]
    fluid = np.flatnonzero(b < 1)  # outside the body
    rng = np.random.default_rng(run)
    k = np.sort(rng.choice(fluid, min(n, fluid.size), replace=False))
    ijk = np.unravel_index(k, b.shape)

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
    """Draw a voxel sample from each store and concatenate them.

    Parameters
    ----------
    out : str or pathlib.Path
        The folder of stores.
    per_run : int
        Number of voxels drawn per store, see ``sample_run``.
    max_runs : int or None, optional
        Sample only the first this many stores, in run order. ``None`` samples all.
    workers : int, optional
        Number of stores sampled at the same time.

    Returns
    -------
    dict
        ``x`` (input channels), ``ing`` (ingredients) and ``y`` (labels), each with the voxels of all
        stores along the last axis; ``channels`` and ``classes`` (names); and ``ncase`` (number of stores).
    """
    paths = stores(out)
    runs = sorted(paths)[:max_runs]
    print(f"sampling {per_run} fluid voxels from each of {len(runs)} stores", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda r: sample_run(r, paths[r], per_run), runs))
    x = np.concatenate([p[0] for p in parts], 1)
    ing = np.concatenate([p[1] for p in parts], 1)
    y = np.concatenate([p[2] for p in parts], 1)
    return dict(x=x, ing=ing, y=y, channels=parts[0][3], classes=parts[0][4], ncase=len(runs))


NAMES = {"u": "u / U∞", "v": "v / U∞", "w": "w / U∞", "Cp": "C_p",
         "body_proximity": "body proximity", "ground_proximity": "ground proximity"}
LIMS = {"u": (-0.6, 1.5), "v": (-0.6, 0.6), "w": (-0.6, 0.6), "Cp": (-1.2, 1.05),
        "body_proximity": (0.0, 1.0), "ground_proximity": (0.0, 1.0)}


def fig_hist(s, fig_dir):
    """Plot the distribution of each input channel, by label class.

    One panel per channel, with a density histogram per class and for the voxels without a label, on a
    log scale. Values are clipped to the channel's range in ``LIMS``. Writes ``feature_hist_by_class.png``.

    Parameters
    ----------
    s : dict
        The sample from ``load_sample``.
    fig_dir : str or pathlib.Path
        The folder the figure is written to.

    Raises
    ------
    KeyError
        If a channel is missing from ``NAMES`` or ``LIMS``.
    """
    x, y, channels, classes = s["x"], s["y"], s["channels"], s["classes"]
    none = ~y.any(0)
    ncol = 3
    nrow = -(-len(channels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4 * nrow), layout="constrained", squeeze=False)
    for ax, (f, ch) in zip(axes.flat, enumerate(channels)):
        lo, hi = LIMS[ch]
        bins = np.linspace(lo, hi, 121)
        ax.hist(np.clip(x[f, none], lo, hi), bins=bins, density=True, histtype="step", color=FREE, linewidth=2,
                linestyle="--", label="no label")
        for i, c in enumerate(classes):
            ax.hist(np.clip(x[f, y[i]], lo, hi), bins=bins, density=True, histtype="step", color=COLOR[c],
                    linewidth=2, label=LABEL[c])
        ax.set_yscale("log")
        ax.set_xlabel(f"{NAMES[ch]} (clipped to [{lo}, {hi}])")
        ax.set_ylabel("probability density (log)")
        ax.set_title(f"{NAMES[ch]} by label class")
        ax.grid(axis="y")
    for ax in axes.flat[len(channels):]:
        ax.set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(h))
    fig.suptitle(f"Input feature distributions per class "
                 f"({x.shape[1]/1e6:.1f}M random fluid voxels from {s['ncase']} runs)")
    fig.savefig(Path(fig_dir) / "feature_hist_by_class.png", dpi=150)
    plt.close(fig)


def fig_corr(s, fig_dir):
    """Plot the Spearman rank correlation of the inputs, the ingredients and the labels.

    Uses a random subsample of at most 2M voxels (seed 0). A column that is constant over the subsample,
    such as a class that never occurs, gives NaN. Writes ``correlation_heatmap.png`` and
    ``correlation_heatmap.csv``.

    Parameters
    ----------
    s : dict
        The sample from ``load_sample``.
    fig_dir : str or pathlib.Path
        The folder the figure and the table are written to.

    Raises
    ------
    KeyError
        If a channel is missing from ``NAMES``.
    """
    from scipy.stats import spearmanr
    x, ing, y, channels, classes = s["x"], s["ing"], s["y"], s["channels"], s["classes"]
    rng = np.random.default_rng(0)
    k = rng.choice(x.shape[1], min(2_000_000, x.shape[1]), replace=False)
    cols = [x[i, k] for i in range(len(channels))] + [ing[i, k] for i in range(len(INGREDIENTS))] \
        + [y[i, k].astype(np.float32) for i in range(len(classes))]
    names = [NAMES[c].replace(" ", "") for c in channels] + [INGREDIENT_NAMES[n] for n in INGREDIENTS] \
        + [LABEL[c] for c in classes]
    r = spearmanr(np.stack(cols, 1)).statistic
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(r, cmap=DIVERGING.reversed(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.spines[:].set_visible(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{r[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color=INK if abs(r[i, j]) < 0.6 else SURF)
    for p in [len(channels) - 0.5, len(channels) + len(INGREDIENTS) - 0.5]:
        ax.axhline(p, color=SURF, linewidth=3); ax.axvline(p, color=SURF, linewidth=3)
    ax.set_xlabel("inputs  |  label ingredients  |  labels")
    ax.set_ylabel("labels  |  label ingredients  |  inputs")  # rotated label reads bottom-to-top
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Spearman rank correlation")
    ax.set_title(f"Spearman correlation ({len(k)/1e6:.1f}M random fluid voxels, {s['ncase']} runs)")
    fig.tight_layout()
    fig.savefig(Path(fig_dir) / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    np.savetxt(Path(fig_dir) / "correlation_heatmap.csv", r, delimiter=",", header=",".join(names), fmt="%.4f")


def main(argv=None):
    """Run the command line.

    Parameters
    ----------
    argv : list[str] or None, optional
        The arguments. ``None`` means ``sys.argv[1:]``.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figs", nargs="+", choices=["slant", "sections", "hist", "corr"])
    ap.add_argument("--out", required=True, help="folder of ahmedml_run<N>.zarr stores")
    ap.add_argument("--fig", required=True, help="where the figures are written")
    ap.add_argument("--runs", type=int, nargs="+", default=[420, 346],
                    help="runs shown by sections (default: %(default)s)")
    ap.add_argument("--per-run", type=int, default=20_000,
                    help="fluid voxels sampled per store for hist and corr (default: %(default)s)")
    ap.add_argument("--max-runs", type=int, default=None,
                    help="sample only the first this many stores (default: %(default)s)")
    ap.add_argument("--workers", type=int, default=8, help="stores sampled at once (default: %(default)s)")
    a = ap.parse_args(argv)
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
