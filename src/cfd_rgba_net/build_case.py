"""
Build one training case: model inputs, label ingredients and labels on the voxel grid.

    cfd-rgba-net-build-case --dataset ahmedml --run 1 --data /data/.../ahmedML/data --out out

Writes one Zarr v3 store, <out>/<tag>.zarr, with <tag> = <dataset>_run<N>:
    inputs          (6, nx, ny, nz) float32: u, v, w, Cp, body_proximity, ground_proximity
    labels          (2, nx, ny, nz) uint8: vortex, shear_layer
    ingredients/    Q_nd, lambda2_nd, omega_mag_nd (float32), core (bool): teacher only
    attributes      grid, case config, contract, label rules, geometry, checks, provenance
and, with --save-vti, <tag>_{inputs,ingredients,labels}.vti for viewing.
The grid fills the dataset's domain (configs/datasets/<name>.yaml `domain`, default: the mesh bounds).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

# Only the standard library at module level: NumPy, SciPy and VTK size their thread pools (OpenBLAS, MKL,
# OpenMP) when first imported, so main() calls setup_threads before anything imports them.
if TYPE_CHECKING:
    from .config import CaseConfig, Contract, LabelRules


def case_tag(cfg: CaseConfig, run: int) -> str:
    """Name one case, the stem of its store and logs.

    Parameters
    ----------
    cfg : CaseConfig
        The dataset configuration.
    run : int
        The run number.

    Returns
    -------
    str
        ``<dataset>_run<run>`` with the dataset name in lower case.
    """
    return f"{cfg.dataset.lower()}_run{run}"


def setup_threads(n: int | None = None) -> int:
    """Set the thread counts of OpenMP, MKL, OpenBLAS and VTK's SMP tools.

    The environment variables are set only if they are not set already, and only take effect if this
    runs before NumPy, SciPy or VTK are first imported.

    Parameters
    ----------
    n : int or None, optional
        Number of threads. ``None`` means all cores.

    Returns
    -------
    int
        The number of threads.
    """
    n = n or os.cpu_count() or 1
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    from vtkmodules.vtkCommonCore import vtkSMPTools
    vtkSMPTools.SetBackend("STDThread")
    vtkSMPTools.Initialize(n)
    return n


def read_geo_parameters(path: Path) -> dict[str, float]:
    """Read a run's geometry parameters from its CSV file.

    Parameters
    ----------
    path : pathlib.Path
        A CSV file with a header row and one row of numbers.

    Returns
    -------
    dict[str, float]
        The value of each column by its stripped header, or an empty dict if the file does not exist.

    Raises
    ------
    ValueError
        If a value is not a number.
    """
    if not path.exists():
        return {}
    with path.open() as fh:
        rows = list(csv.reader(fh))
    return {k.strip(): float(v) for k, v in zip(rows[0], rows[1])}


def build_case(cfg: CaseConfig, run: int, data: Path, out: Path, contract: Contract, rules: LabelRules,
               save_vti: bool = False, log=print, max_voxels: int = 300_000_000) -> dict:
    """Build one training case and write it as a Zarr store.

    The steps: read the volume mesh, select and non-dimensionalize its fields, read the body, build the
    voxel grid, compute the label ingredients on the native mesh, resample them to the grid, assemble the
    input channels, make the labels, and write ``<out>/<dataset>_run<run>.zarr`` (and, with ``save_vti``,
    three ``.vti`` files for viewing).

    Parameters
    ----------
    cfg : CaseConfig
        The dataset configuration.
    run : int
        The run number, which replaces ``{run}`` in the dataset's file paths.
    data : pathlib.Path
        The data root that the file paths are relative to.
    out : pathlib.Path
        The output directory. Created if it does not exist.
    contract : Contract
        The input contract.
    rules : LabelRules
        The label rules.
    save_vti : bool, optional
        Also write ``<tag>_{inputs,ingredients,labels}.vti``.
    log : callable, optional
        Called with one progress message (str) per step. Defaults to ``print``.
    max_voxels : int, optional
        Refuse a grid with more voxels than this.

    Returns
    -------
    dict
        The case metadata that went into the store's attributes, plus ``grid``.

    Raises
    ------
    ValueError
        If the contract's channels differ from the ones ``preprocess`` builds; if ``flow.nu`` is not
        positive; if the grid exceeds ``max_voxels``; or if no voxel lies inside the body.
    FileNotFoundError
        If the volume or body file is missing.
    KeyError
        If the mesh lacks the configured velocity or pressure (see ``prepare_mesh``).
    """
    # imported here, not at module level: see the note on the imports above
    from vtkmodules.vtkCommonCore import vtkSMPTools

    from . import io
    from .config import axis_of
    from .geometry import case_geometry, read_body
    from .ingredients import INGREDIENTS, add_ingredients
    from .label import CLASS_NAMES, describe, make_labels
    from .preprocess import CP, INPUT_CHANNELS, U_ND, assemble_inputs, prepare_mesh, read_volume, resample

    if tuple(contract.channels) != INPUT_CHANNELS:
        raise ValueError(f"contract channels {list(contract.channels)} != the channels preprocess builds "
                         f"{list(INPUT_CHANNELS)}")
    # Re_H = U_inf H / nu goes into the metadata, after all the work: fail before reading anything
    if not cfg.flow.nu > 0:
        raise ValueError(f"flow.nu must be > 0, got {cfg.flow.nu}")
    t0 = time.time()

    def step(msg):
        log(f"[{time.time() - t0:6.0f}s] {msg}")

    out.mkdir(parents=True, exist_ok=True)
    tag = case_tag(cfg, run)
    volume_path = cfg.path("volume", data, run)
    body_path = cfg.path("body", data, run)
    geo_path = cfg.path("geo_parameters", data, run) if "geo_parameters" in cfg.files else None

    check_names = tuple(n for n in (cfg.check_vorticity, cfg.check_cp) if n)
    ug = read_volume(volume_path, cfg.field_names() | set(check_names))
    step(f"read {volume_path.name}: {ug.GetNumberOfCells():,} cells, {ug.GetNumberOfPoints():,} points")
    prep = prepare_mesh(ug, cfg, keep_checks=check_names)
    del ug
    up = axis_of(cfg.flow.up_direction, "up direction")
    domain = cfg.domain or prep.mesh.GetBounds()
    ref = "reference probe " + str(cfg.flow.reference_probe) if cfg.flow.reference_probe else "config"
    step(f"prepared mesh: velocity from {prep.velocity_source}, pressure from {prep.pressure_source}, "
         f"U_inf = {prep.U_inf:.6g}, p_inf = {prep.p_inf:.6g} ({ref}), up {up}, ground at {'xyz'[up.index]} = {prep.ground:.6g}, "
         f"domain {'(config)' if cfg.domain else '(mesh bounds)'} {tuple(round(v, 6) for v in domain)}")

    body = read_body(body_path, cfg.body_scale, cfg.body_merge_tolerance)
    geo = case_geometry(body, prep.ground, cfg.reference_length, up)
    spec = geo.image_spec(contract, domain, cfg.floor_offset_H, cfg.grid_inset_H)
    n_vox = spec.dims[0] * spec.dims[1] * spec.dims[2]
    step(f"H = {geo.H:.6g}, body bounds {tuple(round(b, 6) for b in geo.body_bounds)}, "
         f"grid {spec.dims} ({n_vox:,} voxels), dx = {spec.dx:.6g}")
    if n_vox > max_voxels:
        raise ValueError(f"the domain gives {n_vox:,} voxels (> {max_voxels:,}); set `domain` in the dataset "
                         "config to the region of interest, or raise --max-voxels")

    checks = add_ingredients(prep.mesh, geo.H, cfg.check_vorticity, cfg.check_cp)
    step(f"ingredients ready: {checks}")

    # keep only what is resampled: the probe carries every array along
    keep = {U_ND, CP, *INGREDIENTS}
    prep.mesh.GetCellData().Initialize()
    pd = prep.mesh.GetPointData()
    for name in [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]:
        if name not in keep:
            pd.RemoveArray(name)
    image = resample(prep.mesh, spec, [U_ND, CP, *INGREDIENTS])
    del prep.mesh  # free the mesh; the reference values on prep are still used below
    step("resampled to the voxel grid")

    inp = assemble_inputs(image, body, geo, spec, contract)
    checks["voxels"] = inp.checks
    step(f"inputs assembled: {inp.checks}")
    if inp.checks["n_not_in_mesh_nor_body"] > 0:
        step(f"[warn] {inp.checks['n_not_in_mesh_nor_body']:,} voxels are outside both the mesh and the body "
             "(box leaves the CFD domain, or the mesh has holes); they are not fluid")

    ing = {k: image[k] for k in INGREDIENTS}
    lab = make_labels(ing, inp.fluid, spec, geo, rules, contract.n_per_H)
    step("class fractions over fluid voxels: " + ", ".join(f"{k}={v:.4f}" for k, v in lab.fractions.items()))

    if save_vti:
        io.save_vti(out / f"{tag}_inputs.vti", spec, {"U_nd": inp.inputs[:3], "Cp": inp.inputs[3],
                                                     "body_proximity": inp.inputs[4],
                                                     "ground_proximity": inp.inputs[5]})
        io.save_vti(out / f"{tag}_ingredients.vti", spec, {**ing, "core": lab.core.astype("uint8")})
        io.save_vti(out / f"{tag}_labels.vti", spec, {**{c: lab.labels[i] for i, c in enumerate(CLASS_NAMES)},
                                                     "n_labels": lab.labels.sum(0).astype("uint8")})
        step("wrote vti")

    meta = dict(
        tag=tag,
        dataset=cfg.dataset,
        run=run,
        case_config=cfg.to_dict(),
        contract=contract.to_dict(),
        label_rules=dict(thresholds=rules.to_dict(), text=describe(rules, contract.n_per_H)),
        class_names=list(CLASS_NAMES),
        class_fractions_fluid=lab.fractions,
        geometry=dict(H=geo.H, body_bounds=list(geo.body_bounds), ground=geo.ground,
                      up_axis=str(up), domain=list(domain),
                      Re_H=prep.U_inf * geo.H / cfg.flow.nu,
                      geo_parameters=read_geo_parameters(geo_path) if geo_path else {}),
        channel_names=list(contract.channels),
        reference=dict(U_inf=prep.U_inf, p_inf=prep.p_inf, rho=cfg.flow.rho,
                       source=f"probe at {list(cfg.flow.reference_probe)}" if cfg.flow.reference_probe else "config"),
        fields=dict(velocity=prep.velocity_source, pressure=prep.pressure_source),
        checks=checks,
        provenance=io.provenance(dict(volume=volume_path, body=body_path,
                                      case_config=Path(cfg.source) if cfg.source else None)),
        vtk_threads=vtkSMPTools.GetEstimatedNumberOfThreads(),
        runtime_s=time.time() - t0,
    )
    io.save_case_zarr(out / f"{tag}.zarr", spec,
                      arrays=dict(inputs=inp.inputs, labels=lab.labels),
                      groups=dict(ingredients={**{k: v.astype("float32") for k, v in ing.items()}, "core": lab.core}),
                      attrs=meta)
    step(f"wrote {tag}.zarr")
    return dict(grid=spec.to_dict(), **meta)


def main(argv: list[str] | None = None) -> None:
    """Run the ``cfd-rgba-net-build-case`` command line.

    Parameters
    ----------
    argv : list[str] or None, optional
        The arguments. ``None`` means ``sys.argv[1:]``.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset config: a name in configs/datasets/ or a YAML path")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--data", type=Path, required=True, help="data root holding run_<N>/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-vti", action="store_true", help="also write .vti files for viewing")
    ap.add_argument("--threads", type=int, default=None, help="threads for VTK, OpenMP, MKL and OpenBLAS (default: all cores)")
    ap.add_argument("--max-voxels", type=int, default=300_000_000, help="refuse grids larger than this")
    args = ap.parse_args(argv)
    setup_threads(args.threads)
    from .config import CaseConfig, Contract, LabelRules
    build_case(CaseConfig.load(args.dataset), args.run, args.data, args.out, Contract.load(), LabelRules.load(),
               args.save_vti, log=lambda m: print(m, flush=True), max_voxels=args.max_voxels)


if __name__ == "__main__":
    main()
