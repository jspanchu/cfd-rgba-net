"""
Build one training case: model inputs, label ingredients and labels on the voxel grid.

    cfd-rgba-net-build-case --dataset ahmedml --run 1 --data /data/.../ahmedML/data --out out

Writes one Zarr v3 store, <out>/<tag>.zarr, with <tag> = <dataset>_run<N>:
    inputs          (5, nx, ny, nz) float32: u, v, w, Cp, sdf/H
    fluid           (nx, ny, nz) bool
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

# Arrays some datasets carry, used only for sanity checks when present.
DATASET_VORTICITY = "vorticityMean"
DATASET_CP = "static(p)_coeffMean"


def setup_threads(n: int | None = None):
    """
    Thread counts for OpenMP / MKL / OpenBLAS (environment, unless already set) and VTK's SMP tools. The
    environment only takes effect if this runs before NumPy, SciPy or VTK are first imported.
    """
    n = n or os.cpu_count() or 1
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    from vtkmodules.vtkCommonCore import vtkSMPTools
    vtkSMPTools.SetBackend("STDThread")
    vtkSMPTools.Initialize(n)
    return n


def read_geo_parameters(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = list(csv.reader(path.open()))
    return {k.strip(): float(v) for k, v in zip(rows[0], rows[1])}


def build_case(cfg: CaseConfig, run: int, data: Path, out: Path, contract: Contract, rules: LabelRules,
               save_vti: bool = False, log=print, max_voxels: int = 300_000_000) -> dict:
    # imported here, not at module level: see the note on the imports above
    from vtkmodules.vtkCommonCore import vtkSMPTools

    from . import io
    from .geometry import case_geometry, read_body
    from .ingredients import INGREDIENTS, add_ingredients
    from .label import CLASS_NAMES, describe, make_labels
    from .preprocess import CP, U_ND, assemble_inputs, prepare_mesh, read_volume, resample

    t0 = time.time()

    def step(msg):
        log(f"[{time.time() - t0:6.0f}s] {msg}")

    out.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.dataset.lower()}_run{run}"
    volume_path = cfg.path("volume", data, run)
    body_path = cfg.path("body", data, run)
    geo_path = cfg.path("geo_parameters", data, run) if "geo_parameters" in cfg.files else None

    ug = read_volume(volume_path, cfg.field_names() | {DATASET_VORTICITY, DATASET_CP})
    step(f"read {volume_path.name}: {ug.GetNumberOfCells():,} cells, {ug.GetNumberOfPoints():,} points")
    prep = prepare_mesh(ug, cfg, keep_checks=(DATASET_VORTICITY, DATASET_CP))
    del ug
    up = cfg.flow.up_axis
    domain = cfg.domain or prep.mesh.GetBounds()
    ref = "reference probe " + str(cfg.flow.reference_probe) if cfg.flow.reference_probe else "config"
    step(f"prepared mesh: velocity from {prep.velocity_source}, pressure from {prep.pressure_source}, "
         f"U_inf = {prep.U_inf:.6g}, p_inf = {prep.p_inf:.6g} ({ref}), up {up}, ground at {'xyz'[up.index]} = {prep.ground:.6g}, "
         f"domain {'(config)' if cfg.domain else '(mesh bounds)'} {tuple(round(v, 6) for v in domain)}")

    body = read_body(body_path, cfg.body_scale)
    geo = case_geometry(body, prep.ground, cfg.reference_length, up)
    spec = geo.image_spec(contract, domain, cfg.floor_offset_H)
    n_vox = spec.dims[0] * spec.dims[1] * spec.dims[2]
    step(f"H = {geo.H:.6g}, body bounds {tuple(round(b, 6) for b in geo.body_bounds)}, "
         f"grid {spec.dims} ({n_vox:,} voxels), dx = {spec.dx:.6g}")
    if n_vox > max_voxels:
        raise ValueError(f"the domain gives {n_vox:,} voxels (> {max_voxels:,}); set `domain` in the dataset "
                         "config to the region of interest, or raise --max-voxels")

    checks = add_ingredients(prep.mesh, geo.H, DATASET_VORTICITY, DATASET_CP)
    step(f"ingredients ready: {checks}")

    # keep only what is resampled: the probe carries every array along
    keep = {U_ND, CP, *INGREDIENTS}
    prep.mesh.GetCellData().Initialize()
    pd = prep.mesh.GetPointData()
    for name in [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]:
        if name not in keep:
            pd.RemoveArray(name)
    image = resample(prep.mesh, spec, [U_ND, CP, *INGREDIENTS])
    del prep.mesh
    step("resampled to the voxel grid")

    inp = assemble_inputs(image, body, geo, spec)
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
                                                     "sdf_H": inp.inputs[4], "fluid": inp.fluid.astype("uint8")})
        io.save_vti(out / f"{tag}_ingredients.vti", spec, {**ing, "core": lab.core.astype("uint8")})
        io.save_vti(out / f"{tag}_labels.vti", spec, {**{c: lab.labels[i] for i, c in enumerate(CLASS_NAMES)},
                                                     "n_labels": lab.labels.sum(0).astype("uint8")})
    if save_vti:
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
        provenance=io.provenance(dict(volume=volume_path, body=body_path, case_config=Path(cfg.source or ""))),
        vtk_threads=vtkSMPTools.GetEstimatedNumberOfThreads(),
        runtime_s=time.time() - t0,
    )
    io.save_case_zarr(out / f"{tag}.zarr", spec,
                      arrays=dict(inputs=inp.inputs, fluid=inp.fluid, labels=lab.labels),
                      groups=dict(ingredients={**{k: v.astype("float32") for k, v in ing.items()}, "core": lab.core}),
                      attrs=meta)
    step(f"wrote {tag}.zarr")
    return dict(grid=spec.to_dict(), **meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset config: a name in configs/datasets/ or a YAML path")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--data", type=Path, required=True, help="data root holding run_<N>/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-vti", action="store_true", help="also write .vti files for viewing")
    ap.add_argument("--threads", type=int, default=None, help="VTK SMP threads (default: all cores)")
    ap.add_argument("--max-voxels", type=int, default=300_000_000, help="refuse grids larger than this")
    args = ap.parse_args(argv)
    setup_threads(args.threads)
    from .config import CaseConfig, Contract, LabelRules
    build_case(CaseConfig.load(args.dataset), args.run, args.data, args.out, Contract.load(), LabelRules.load(),
               args.save_vti, log=lambda m: print(m, flush=True), max_voxels=args.max_voxels)


if __name__ == "__main__":
    main()
