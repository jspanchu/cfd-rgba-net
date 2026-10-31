"""
Input is one AhmedML run. Output is three vtkImageData volumes (+ .npz) for training:

    <tag>_inputs.vti        INPUTS              U_{nd}:(u,v,w)/U_{inf}, C_p, mask           (network input X)
    <tag>_ingredients.vti   LABEL INGREDIENTS   Q_{nd}, lambda2_{nd}, omega_mag_{nd},C_{pt},
                                                reversed_flow_frac, mask_eroded             (teacher only)
    <tag>_labels.vti        GROUND TRUTH        recirculation, vortex, shear-layer, wake    (0/1 per voxel)

    Pipeline:
        1. read the native unstructured grid (*.vtu): UMean, pMean
        2. compute the label ingredients on the native mesh
            Q, lambda2, |omega| <- S, W <- grad(U) via vtkGradientFilter cell-centered.
            C_{pt} = C_p + |u|^2/U_{inf}^{2}, reversed = (u/Uinf < -1e-6)
        3. resample the inputs and ingredients together onto a uniform image, dx = H/n_per_H
        4. apply the four label rules on the image to generate ground truth
        5. write inputs/ingredients/labels as separate .vti, and .npz, plus a metadata .json

    Non-dimensionalization:
        1. x* = x/H
        2. u* = u/U_{inf}
        3. Q* = Q * H^2 / U_{inf} ^ 2
        4. lambda2* = lambda2 * H^2 / U_{inf}^2
        5. |omega|* = |omega| * H / U_{inf}
        6. C_p = 2 * (p - 0)    (rho = 1, and U_{inf}=1)

    Label rules v1:
        Constants:
            tau_{lambda2}   : negated and used as upper bound on lambda2* to determine vortex
            tau_{omega}     : used as lower bound on |omega|* to determine shear layer
            tau_{cpt}       : used as upper bound on C_{pt} to determine wake
            min_cc          : used for recirculation to drop connected components with # voxels < min_cc voxels
            floor_exclude_H : exclude z < floor_exclude_H * H from vortex/shear/wake labels (stationary wind-tunnel floor BL); 0 disables it
            box             : resample box dimensions [-up, down] x [-lat, +lat] x [0, top]

        Decision rules:
            recirculation : reversed_flow_frac > reversed_frac, then connected copmonents with < min_cc voxels removed
            vortex        : lambda2* < -tau_{lambda2}  AND Q* > 0              (on the wall-eroded mask)
            shear_layer   : |omega|* > tau_{omega}     AND NOT vortex          (on the wall-eroded mask)
            wake          : C_{pt} < tau_{cpt}         AND x > x_{base}        (on the wall-eroded mask)

    Usage:
        python build_training_case.py --run 23 --n-per-H 32 --out out
"""

import argparse
import csv
import numpy as np
import os
import time
import json
import scipy.ndimage as ndi
from scipy.stats import spearmanr

from pathlib import Path

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkAcceleratorsVTKmFilters import vtkmFilterOverrides
from vtkmodules.vtkCommonCore import vtkSMPTools
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid, vtkDataObject, vtkImageData
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData, vtkResampleToImage
from vtkmodules.vtkFiltersGeneral import vtkGradientFilter
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader, vtkXMLImageDataWriter

vtkSMPTools.SetBackend("STDThread")
vtkmFilterOverrides.SetEnabled(False)

if nthreads:=os.cpu_count():
    vtkSMPTools.Initialize(nthreads)
print(vtkSMPTools.GetBackend())
print(f"Number of threads: {vtkSMPTools.GetEstimatedNumberOfThreads()}")
# Set the maximum number of threads for all common BLAS backends
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)

def log(msg, t0):
    print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)

def read(file_name: str | os.PathLike):
    suffix = Path(file_name).suffix
    if suffix == ".stl":
        reader = vtkSTLReader(file_name=file_name);
        reader.Update()
        return reader.output
    elif suffix == ".vtu":
        reader = vtkXMLUnstructuredGridReader(file_name=file_name);
        reader.Update()
        return reader.output
    else:
        raise RuntimeError()

def eigen_values_lambda2(G, n_chunk=2_000_000):
    """
    G: (N, 3, 3) velocity gradient tensor
    Returns lambda2 of S^2 + Omega^2. Shape (N,)
    """
    N = G.shape[0]
    out = np.empty(N, dtype=np.float32)
    for i in range(0, N, n_chunk):
        # upcast for accuracy because M subtracts terms of similar size and lam2 is middle
        # eigen val often close to 0.
        g = G[i:i+n_chunk].astype(np.float64)
        S = 0.5 * (g + np.transpose(g, (0, 2, 1)))
        W = 0.5 * (g - np.transpose(g, (0, 2, 1)))
        M = S @ S + W @ W
        out[i:i+n_chunk] = np.linalg.eigvalsh(M)[:, 1]
    return out

def ingredients(input: vtkUnstructuredGrid, pName: str, UName: str, Uinf, H, p0, rho, wall_tolerance, t0):
    grad_filter = vtkGradientFilter()
    grad_filter.compute_divergence = False
    grad_filter.compute_q_criterion = False
    grad_filter.compute_vorticity = False
    grad_filter.input_data = input
    grad_filter.result_array_name = "gradU"
    grad_filter.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_CELLS, UName)
    grad_filter.Update()
    log("gradU per cell ready", t0)

    G = np.asarray(grad_filter.output.cell_data["gradU"]).reshape(-1, 3, 3)
    grad_filter.output.cell_data.RemoveArray("gradU")
    scale = H / Uinf
    G  = (G * scale).astype(np.float32)
    bad = ~np.isfinite(G).all(axis=(1, 2))
    n_bad = int(bad.sum())
    G[bad] = 0.0
    log(f"non-finite gradient at {n_bad} of {G.shape[0]:,} cells set to 0", t0)

    S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    W = 0.5 * (G - np.transpose(G, (0, 2, 1))) # Omega
    Q = 0.5 * (np.sum(W**2, axis=(1, 2)) - np.sum(S**2, axis=(1, 2))).astype(np.float32)
    del S,W
    wx = (G[:, 2, 1] - G[:, 1, 2])
    wy = (G[:, 0, 2] - G[:, 2, 0])
    wz = (G[:, 1, 0] - G[:, 0, 1])
    omega_mag = np.sqrt(wx**2 + wy**2 + wz**2).astype(np.float32)
    log("Q, |omega| per cell ready", t0)
    lambda2 = eigen_values_lambda2(G)
    del G
    log("lambda2 per cell ready", t0)

    checks = {"n_cells_nonfinite_gradient": n_bad}
    # compare |omega| against the dataset's own (OpenFOAM) vorticityMean.
    # spearman_r checks the spatial pattern (rank agreement), median_ratio checks the scale.
    # cells below the median are mostly free stream where both are ~0 and noisy.
    if "vorticityMean" in input.cell_data.keys():
        omega_mag_native = np.linalg.norm(np.asarray(input.cell_data["vorticityMean"]), axis=1) * scale # same scaling as our omega_mag_nd
        # pick atleast 2M at random or all cells.
        rng = np.random.default_rng(0)
        idx = rng.choice(omega_mag_native.size, min(2_000_000, omega_mag_native.size), replace=False)
        sel = idx[omega_mag_native[idx] > np.median(omega_mag_native)] # ignore cells with near-zero omega
        checks["omega_vs_dataset_vorticityMean"] = dict(
            spearman_r=float(spearmanr(omega_mag[sel], omega_mag_native[sel]).statistic),
            median_ratio=float(np.median(omega_mag[sel] / np.maximum(omega_mag_native[sel], 1e-12))),
            note="per-cell, 2M-cell random subsample, cells above median dataset |omega|",
        )
        log(f"check |omega| vs dataset vorticityMean: {checks['omega_vs_dataset_vorticityMean']}", t0)

    # Distribute the derived quantities to the points
    c2p = vtkCellDataToPointData(pass_cell_data=False)
    cell_input = vtkUnstructuredGrid()
    cell_input.ShallowCopy(grad_filter.output)
    cell_input.point_data.Initialize()
    cell_input.cell_data.Initialize()
    cell_input.cell_data["lambda2_nd"] = lambda2
    cell_input.cell_data["Q_nd"] = Q
    cell_input.cell_data["omega_mag_nd"] = omega_mag
    c2p.input_data = cell_input
    c2p.Update()

    # Pointwise quantities from the point data inputs.
    U_nd = input.point_data[UName].astype(np.float32) / Uinf
    p = input.point_data[pName].astype(np.float32)
    Cp = 2.0 * (p - p0) / (rho * Uinf**2)
    Cpt = Cp + np.sum(U_nd * U_nd, axis=1)
    if "static(p)_coeffMean" in input.point_data:
        checks["Cp_vs_dataset_static_p_coeffMean_maxabs"] = float(np.abs(Cp - np.asarray(input.point_data["static(p)_coeffMean"])).max())

    output = vtkUnstructuredGrid()
    output.ShallowCopy(c2p.input)
    output.cell_data.Initialize()
    output.point_data.Initialize()
    output.point_data["U_nd"] = U_nd
    output.point_data["Cp"] = Cp
    output.point_data["Cpt"] = Cpt.astype(np.float32)
    output.point_data["reversed_flow"] = (U_nd[:, 0] < wall_tolerance).astype(np.float32)
    for k in ("Q_nd", "lambda2_nd", "omega_mag_nd"):
        output.point_data[k] = np.asarray(c2p.output.point_data[k], dtype=np.float32)
    return output, checks


def resample(vol: vtkUnstructuredGrid, dims: tuple[int, int, int], bx0, bx1, by0, by1, bz0, bz1):
    resampler = vtkResampleToImage(
        input_data_object=vol,
        sampling_dimensions=dims,
        sampling_bounds=(bx0, bx1, by0, by1, bz0, bz1),
        use_input_bounds=False,
    )
    resampler.Update()
    return resampler.output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1, help="run number")
    ap.add_argument("--data", default="./data", help="data directory")
    ap.add_argument("--out", default="./out", help="output directory")
    ap.add_argument("--n-per-H", type=int, default=32, help="voxels per H")
    ap.add_argument("--uinf", type=float, default=1.0, help="freestream velocity")
    ap.add_argument("--pinf", type=float, default=0.0, help="farfield pressure")
    ap.add_argument("--rho", type=float, default=1.0, help="fluid density")
    ap.add_argument("--nu", type=float, default=3.75e-7, help="fluid kinematic visc")
    ap.add_argument("--up", type=float, default=0.5, help="position of the upstream face of box in body lengths")
    ap.add_argument("--down", type=float, default=2.0, help="position of the downstream face of box in body lengths")
    ap.add_argument("--lat", type=float, default=2.0, help="y-position of the lateral face of the box in body widths, other y-face is at -ve this value")
    ap.add_argument("--top", type=float, default=2.5, help="z-position of the top face of box in body heights, bottom is 0")
    # thresholds
    ap.add_argument("--tau-l2", type=float, default=0.5, help="vortex: lambda2* < -tau_{lambda2} AND Q* > 0")
    ap.add_argument("--tau-om", type=float, default=2.0, help="shear_layer: |omega|* > tau_{omega} AND NOT vortex")
    ap.add_argument("--tau-cpt", type=float, default=0.8, help="wake: C_{pt} < tau_{cpt} AND x > x_{base}")
    ap.add_argument("--min_cc", type=int, default=8, help="recirculation: drop components < min_cc voxels")
    ap.add_argument("--reversed-frac", type=float, default=0.5,
        help="used in determining cells with reversed flow > frac due to resample filter interpolating between 0 and 1")
    ap.add_argument("--floor-exclude-H", type=float, default=0.12,
        help="exclude z < floor_exclude_H*H from vortex/shear/wake labels (stationary wind-tunnel floor BL); 0 disables it")
    ap.add_argument("--wall-tolerance", type=float, default=-1.0e-6,
        help="Decides the mesh points which count as reversed. The floor is no-slip, ideally U=0 but in practice it is ~3e-17.")
    ap.add_argument("--wall-erosion-voxels", type=int, default=1,
        help="how many voxel layers to strip off the walls")
    ap.add_argument("--save-vti", action="store_true", default=False,
        help="Write the .vti files (only for viewing, the .npz have the same data)")
    args = ap.parse_args()
    t0 = time.time()
    run = args.run
    tag = f"ahmed_run{run}_dx_H_by_{args.n_per_H}"
    d = Path(args.data) / f"run_{run}"
    outputs = Path(args.out)
    outputs.mkdir(exist_ok=True, parents=True)

    # ---- geometry -------------------------------- #
    body = read(file_name=d / f"ahmed_{run}.stl")
    xmin, xmax, ymin, ymax, zmin, zmax = body.bounds
    L, W, H = xmax - xmin, ymax - ymin, zmax - zmin
    geo = {}
    gp = d / f"geo_parameters_{run}.csv"
    if gp.exists():
        rows = list(csv.reader(gp.open()))
        geo = {k.strip(): float(v) for k, v in zip(rows[0], rows[1])}
        if abs(geo.get("body-height", H * 1000) / 1000 - H) > 0.02 * H:
            print(f"[warn] geo file's body-height {geo['body-height']/1000:.4f} vs STL {H:.4f} is > 20%")
    dx = H / args.n_per_H
    log(f"geometry L={L:.4f} W={W:.4f} H={H:.4f} m, xmin={xmin:.4f} xmax={xmax:.4f} ymin={ymin:.4f} ymax={ymax:.4f} zmin={zmin:.4f} zmax={zmax:.4f} m, dx={dx:.5f} m (H/{args.n_per_H})", t0)

    # ---- native mesh: inputs & ingredients ------- #
    vol = read(file_name=d / f"volume_{run}.vtu")
    pName = None
    for name in ["pMean", "p"]:
        if name in vol.point_data.keys():
            pName = name
            break
    if pName is None:
        print(f"[err] volume_{run}.vtu does not have 'pMean' or 'p' array in point data")
        exit(1)
    UName = None
    for name in ["UMean", "U"]:
        if name in vol.point_data.keys() and name in vol.cell_data.keys():
            UName = name
            break
    if UName is None:
        print(f"[err] volume_{run}.vtu does not have 'UMean' or 'U' array in point and cell data")
        exit(1)
    # drop unused arrays so the gradient filter and resampler don't carry them along
    for name in list(vol.point_data.keys()):
        if name not in (UName, pName, "static(p)_coeffMean"):
            vol.point_data.RemoveArray(name)
    for name in list(vol.cell_data.keys()):
        if name not in (UName, "vorticityMean"):
            vol.cell_data.RemoveArray(name)
    log(f"volume n_cells={vol.number_of_cells} n_points={vol.number_of_points}", t0)
    vol, checks = ingredients(vol, pName, UName, args.uinf, H, args.pinf, args.rho, args.wall_tolerance, t0)

    # ---- resample to image ----------------------- #
    bx0, bx1 = xmin - args.up * L, xmax + args.down * L
    by0, by1 = -args.lat * W, args.lat * W
    bz0, bz1 = 0.0, args.top * H
    nx = int(np.ceil((bx1 - bx0) / dx)) + 1
    ny = int(np.ceil((by1 - by0) / dx)) + 1
    nz = int(np.ceil((bz1 - bz0) / dx)) + 1
    # adjust end of box to fit nx, ny, nz
    bx1 = bx0 + (nx - 1) * dx
    by1 = by0 + (ny - 1) * dx
    bz1 = bz0 + (nz - 1) * dx
    dims = (nx, ny, nz)
    log(f"box: ({bx0}, {bx1}) x ({by0}, {by1}) x ({bz0}, {bz1})", t0)
    img = resample(vol, dims, bx0, bx1, by0, by1, bz0, bz1)
    log(f"resampled to {dims}", t0)

    def F(name):
        a = np.asarray(img.point_data[name])
        if a.ndim == 1:
            return a.reshape(dims, order='F') # for a[i, j, k] to work
        else:
            channels = []
            for i in range(a.shape[1]):
                channels.append(a[:, i].reshape(dims, order='F'))
            return np.stack(channels, 0) # (C, nx, ny, nz)

    mask = (F("vtkValidPointMask").view(np.uint8) > 0).astype(np.uint8)
    m = mask.astype(bool)
    # exclude voxels that are adjacent to body/floor
    m_eroded = ndi.binary_erosion(m, iterations=args.wall_erosion_voxels)
    # zero out the masked voxels
    U = F("U_nd").astype(np.float32)
    U[:, ~m] = 0.0
    Cp = F("Cp").astype(np.float32)
    Cpt = F("Cpt").astype(np.float32)
    rev = F("reversed_flow").astype(np.float32)
    for a in [Cp, Cpt, rev]:
        a[~m] = 0.0
    # derived quantities come from gradients.
    # near walls, they're dominated by boundary layer
    # and after resampling, they produce large
    # values that would show up as false vortex/shear labels.
    # so zero them at the eroded mask
    Q = F("Q_nd").astype(np.float32)
    lambda2 = F("lambda2_nd").astype(np.float32)
    omega_mag = F("omega_mag_nd").astype(np.float32)
    for a in [Q, lambda2, omega_mag]:
        a[~m_eroded] = 0.0

    # ---- label rules ----------------------------- #
    x = bx0 + dx * np.arange(nx)
    behind_base = (x > xmax)[:, None, None]
    recirc = (rev > args.reversed_frac) & m
    lab, n = ndi.label(recirc)
    if n > 0:
        sizes = ndi.sum_labels(recirc, lab, range(1, n+1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sizes >= args.min_cc
        recirc = keep[lab] # looks up value of keep at l = lab[i,j,k]
    zc = bz0 + dx * np.arange(nz)
    above_floor = (zc >= args.floor_exclude_H * H)[None, None, :]
    vortex = (lambda2 < -args.tau_l2) & (Q > 0) & m_eroded & above_floor
    shear = (omega_mag > args.tau_om) & ~vortex & m_eroded & above_floor
    wake = (Cpt < args.tau_cpt) & behind_base & m_eroded & above_floor
    labels = np.stack([recirc, vortex, shear, wake], axis=0).astype(np.uint8)
    class_names = ["recirculation", "vortex", "shear_layer", "wake"]
    fracs = {c: float(labels[i][m].mean()) for i, c in enumerate(class_names)}
    fracs["free_stream(none)"] = float((labels.sum(0) == 0)[m].mean())
    log("class voxel fractions (fluid voxels): " + ", ".join(f"{k}={v:.4f}" for k, v in fracs.items()), t0)

    # ---- write ----------------------------------- #
    np.savez_compressed(outputs / f"{tag}_inputs.npz",
        inputs=np.concatenate([U, Cp[None], mask[None].astype(np.float32)], axis=0),
        channel_names=np.array(["u/Uinf", "v/Uinf", "w/Uinf", "Cp", "mask"]),
        origin=np.array([bx0, by0, bz0]),
        spacing=dx,
        dims=np.array(dims),
    )
    log(f"wrote {tag}_inputs.npz", t0)
    np.savez_compressed(outputs / f"{tag}_ingredients.npz",
        Q_nd=Q,
        lambda2_nd=lambda2,
        omega_mag_nd=omega_mag,
        Cpt=Cpt,
        reversed_flow_frac=rev,
        mask_eroded=m_eroded.astype(np.uint8),
        origin=np.array([bx0, by0, bz0]),
        spacing=dx,
        dims=np.array(dims),
    )
    log(f"wrote {tag}_ingredients.npz", t0)
    np.savez_compressed(outputs / f"{tag}_labels.npz",
        labels=labels,
        class_names=np.array(class_names),
        origin=np.array([bx0, by0, bz0]),
        spacing=dx,
        dims=np.array(dims),
    )
    log(f"wrote {tag}_labels.npz", t0)

    def put(image, name, np_array):
        image.point_data[name] = numpy_to_vtk((np_array.reshape(-1, order="F") if np_array.ndim == 3
            else np.stack([np_array[i].reshape(-1, order="F") for i in range(np_array.shape[0])], 1)), deep=1)
    def write_image(file_name, np_arrays):
        image = vtkImageData(dimensions=dims, spacing=(dx, dx, dx), origin=(bx0, by0, bz0))
        for name, np_array in np_arrays.items():
            put(image, name, np_array)
        writer = vtkXMLImageDataWriter(file_name=file_name)
        writer.input_data = image
        writer.Write()
    if not args.save_vti:
        write_image(outputs / f"{tag}_inputs.vti", {
            "U_nd": U,
            "Cp": Cp,
            "mask": mask,
        })
        log(f"wrote {tag}_inputs.vti", t0)
        write_image(outputs / f"{tag}_ingredients.vti", {
            "Q_nd": Q,
            "lambda2_nd": lambda2,
            "omega_mag_nd": omega_mag,
            "Cpt": Cpt,
            "reversed_flow_frac": rev,
            "mask_eroded": m_eroded.astype(np.uint8),
        })
        log(f"wrote {tag}_ingredients.vti", t0)
        label_np_array_dict = {
            c: labels[i] for i, c in enumerate(class_names)
        }
        label_np_array_dict.update({"n_labels": labels.sum(0).astype(np.uint8)})
        write_image(outputs / f"{tag}_labels.vti", label_np_array_dict)
        log(f"wrote {tag}_labels.vti", t0)

    meta = dict(
        rules_version="v1-frozen-2026-10-06",
        run=run,
        dataset="AhmedML",
        tag=tag,
        U_array=UName,
        p_array=pName,
        Uinf=args.uinf,
        pinf=args.pinf,
        rho=args.rho,
        nu=args.nu,
        Re_H=args.uinf * H / args.nu,
        L=L,
        W=W,
        H=H,
        x_base=xmax,
        geo_parameters=geo,
        n_per_H=args.n_per_H,
        dx=dx,
        dims=dims,
        origin=[bx0, by0, bz0],
        box_body_units=dict(up_L=args.up, down_L=args.down, lat_W=args.lat, top_H=args.top),
        ingredients_computed_on="VTU mesh, cell-centered vtkGradientFilter -> vtkCellToPointData -> vtkResampleToImage",
        vtk_smp_backend=vtkSMPTools.GetBackend(),
        vtk_threads=vtkSMPTools.GetEstimatedNumberOfThreads(),
        thresholds=dict(
            tau_l2=args.tau_l2,
            tau_om=args.tau_om,
            tau_cpt=args.tau_cpt,
            min_cc=args.min_cc,
            floor_exclude_H=args.floor_exclude_H,
            reversed_frac=args.reversed_frac,
            wall_erosion_voxels=args.wall_erosion_voxels,
            wall_tolerance=args.wall_tolerance,
        ),
        label_rules=dict(
            recirculation=f"reversed_flow_frac>{args.reversed_frac} & mask (reversed = u*<{args.wall_tolerance} on native mesh), components<{args.min_cc} removed",
            vortex=f"lambda2* < -{args.tau_l2} & Q*>0 & mask_eroded & z>={args.floor_exclude_H}*H",
            shear_layer=f"|omega|*>{args.tau_om} & ~vortex & mask_eroded & z>={args.floor_exclude_H}*H",
            wake=f"Cpt<{args.tau_cpt} & x>{xmax} & mask_eroded & z>={args.floor_exclude_H}*H",
        ),
        class_names=class_names,
        class_fractions_fluid=fracs,
        valid_fraction=float(m.mean()),
        validation=checks,
        runtime_s=time.time() - t0,
    )
    json.dump(meta, (outputs / f"{tag}_meta.json").open("w"), indent=2, default=float)
    log(f"wrote {tag}_meta.json", t0)

if __name__ == "__main__":
    main()
