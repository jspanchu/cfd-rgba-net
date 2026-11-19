"""Volume rendering of one unseen case.

The pipeline runs raw volume mesh -> classes on the voxel grid -> one PNG per
view.

    cfd-rgba-net-infer volume_280.vtu --body ahmed_280.stl --method net \
        --checkpoint runs/pilot/best.pt --out out/run280_net
        [--views side top rear] [--tau-lambda2 0.5 --tau-omega 2.0]

--method picks what is drawn, each from the .vtu alone:

    net        the net's probabilities from the grid features (resample, finite
    differences on the grid), squashed with --tau-lambda2 and --tau-omega
    (default: the label rules' thresholds) baseline   the label rules on the
    same grid features labels     the label rules on the ingredients of the
    native mesh (vtkGradientFilter), then resampled

The net also runs on inputs the user has already computed and squashed into
[-1, 1], one channel per name in --arrays, from the point arrays of a .vti on
the voxel grid:

    cfd-rgba-net-infer inputs.vti --arrays a b c \
        --checkpoint runs/pilot/best.pt --out out/mine [--body body.stl]

All of its voxels are scored, and --body is only drawn.

Writes <out>_<view>.png per view and, for net, <out>_<view>_uncertainty.png: the
entropy of the net's three class probabilities over ln 3, drawn alone. With
--body these are close-ups of the body and its near wake, and
<out>_<view>_domain[_uncertainty].png frame the whole grid. The
classes are 0 outside the core (walls and the floor), as in
cfd-rgba-net-evaluate. Without --body, pass --H, the reference length in mesh
units; the body is then the hole in the mesh.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkUnstructuredGrid
from vtkmodules.vtkIOXML import vtkXMLImageDataReader

from .baseline import default_device, grid_features
from .config import CaseConfig, Contract, LabelRules
from .geometry import ImageSpec, body_inside
from .ingredients import add_ingredients, gpu_criteria
from .label import core_mask
from .preprocess import U_ND, prepare_mesh, resample_case


def case_features(
    ug: vtkUnstructuredGrid,
    cfg: CaseConfig,
    contract: Contract,
    rules: LabelRules,
    body: vtkPolyData | None = None,
    H: float | None = None,
) -> tuple[torch.Tensor, ImageSpec, torch.Tensor]:
    """Compute the net's input features of one case on its voxel grid.

    Follows ``build_case``: the same mesh preparation, grid, resampling and
    ``grid_features``. U_inf, the up axis, the ground and the domain come from
    ``cfg`` and the mesh. With a ``body``, H and the inside-body mask come from
    it, as in ``build_case``. Without, the body must be a hole in the mesh: the
    voxels no mesh cell holds are the non-fluid voxels.

    Parameters
    ----------
    ug : vtkUnstructuredGrid
        The raw volume mesh, with the arrays that ``cfg`` names. Left unchanged.
    cfg : CaseConfig
        The dataset configuration.
    contract : Contract
        The input contract. Supplies ``n_per_H``.
    rules : LabelRules
        The label rules. Supply the core.
    body : vtkPolyData or None, optional
        The closed body surface, in mesh units after ``cfg.body_scale`` (see
        ``read_body``).
    H : float or None, optional
        The reference length, used only without a ``body``.
        ``cfg.reference_length`` overrides it.

    Returns
    -------
    features : torch.Tensor
        Float32 ``(3, nx, ny, nz)``: lambda2*, Q* and |omega|*, ready for
        ``evaluate.predict``.
    spec : ImageSpec
        The voxel grid of ``features``.
    core : torch.Tensor
        Bool ``(nx, ny, nz)``: the voxels where labels may be set
        (``label.core_mask``). The net was trained and scored there only, so its
        probabilities elsewhere, near walls and the floor, mean nothing.
    """
    prep = prepare_mesh(ug, cfg)
    grid = resample_case(
        prep,
        cfg,
        contract,
        body,
        names=[U_ND],
        H=H,
        ingredients=None,
    )
    image, spec, geo = grid.image, grid.spec, grid.geo
    # without the surface, the body is the hole in the mesh: no cell holds its
    # voxels
    inside = body_inside(body, spec) if body is not None else ~image["valid"]
    u = image[U_ND].astype(np.float32)
    u[:, ~image["valid"] | inside] = 0.0
    # the gradient mask is "not inside the body", as in build_case, not "fluid"
    features = grid_features(u, ~inside, 1 / contract.n_per_H)
    core = core_mask(
        ~inside, geo.height_above_ground(spec) / geo.H, rules, contract.n_per_H
    )[1]
    return torch.from_numpy(features), spec, torch.from_numpy(core)


def case_ingredients(
    ug: vtkUnstructuredGrid,
    cfg: CaseConfig,
    contract: Contract,
    rules: LabelRules,
    body: vtkPolyData | None = None,
    H: float | None = None,
) -> tuple[torch.Tensor, ImageSpec, torch.Tensor]:
    """Compute the label ingredients on the native mesh, probe them on the grid.

    As in ``build_case``, the gradient comes from ``vtkGradientFilter`` on the
    native mesh. Unlike it, Q, |omega| and lambda2 are computed in torch with
    ``lambda2_closed_form``, so the result may differ from the stored labels by
    float32 round-off near the thresholds.

    Parameters
    ----------
    ug : vtkUnstructuredGrid
        The raw volume mesh, with the arrays that ``cfg`` names. Left unchanged.
    cfg : CaseConfig
        The dataset configuration.
    contract : Contract
        The input contract. Supplies ``n_per_H``.
    rules : LabelRules
        The label rules. Supply the core.
    body : vtkPolyData or None, optional
        The closed body surface, in mesh units after ``cfg.body_scale``.
    H : float or None, optional
        The reference length, used only without a ``body``.

    Returns
    -------
    ingredients : torch.Tensor
        Float32 ``(3, nx, ny, nz)``: lambda2*, Q* and |omega|*, in the order of
        the grid features.
    spec : ImageSpec
        The voxel grid.
    core : torch.Tensor
        Bool ``(nx, ny, nz)``: the voxels where labels may be set, from the
        fluid mask of ``build_case``.
    """
    names = ["lambda2_nd", "Q_nd", "omega_mag_nd"]
    prep = prepare_mesh(ug, cfg)
    grid = resample_case(
        prep,
        cfg,
        contract,
        body,
        names=names,
        H=H,
        ingredients=(
            lambda mesh, h: add_ingredients(mesh, h, criteria=gpu_criteria)
        ),
    )
    image, spec, geo = grid.image, grid.spec, grid.geo
    inside = body_inside(body, spec) if body is not None else ~image["valid"]
    fluid = image["valid"] & ~inside
    core = core_mask(
        fluid, geo.height_above_ground(spec) / geo.H, rules, contract.n_per_H
    )[1]
    return (
        torch.from_numpy(np.stack([image[n] for n in names])),
        spec,
        torch.from_numpy(core),
    )


def read_inputs(path: Path, names: list[str]) -> tuple[torch.Tensor, ImageSpec]:
    """Read the net's inputs from the point arrays of a VTK image file.

    Parameters
    ----------
    path : pathlib.Path
        The ``.vti`` file, with the same spacing along every axis.
    names : list[str]
        The point arrays, in channel order. A vector array gives one channel per
        component.

    Returns
    -------
    inputs : torch.Tensor
        Float32 ``(n_in, nx, ny, nz)``.
    spec : ImageSpec
        The voxel grid of the image.

    Raises
    ------
    ValueError
        If the spacing is not the same along every axis, or an array is
        missing.
    """
    reader = vtkXMLImageDataReader(file_name=str(path))
    reader.Update()
    image = reader.GetOutput()
    spacing = image.GetSpacing()
    if not np.allclose(spacing, spacing[0]):
        raise ValueError(f"{path}: spacing {spacing} is not isotropic")
    dims = image.GetDimensions()
    channels = []
    for name in names:
        a = image.GetPointData().GetArray(name)
        if a is None:
            raise ValueError(f"{path}: no point array {name!r}")
        # x varies fastest, as save_vti writes it
        a = vtk_to_numpy(a).reshape(-1, a.GetNumberOfComponents())
        channels += [c.reshape(dims, order="F") for c in a.T]
    spec = ImageSpec(tuple(image.GetOrigin()), spacing[0], dims)
    return torch.from_numpy(np.stack(channels).astype(np.float32)), spec


def normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    """Measure how uncertain the net is per voxel.

    The measure is the entropy of its class probabilities over its maximum.

    Parameters
    ----------
    p : torch.Tensor
        Float ``(K, nx, ny, nz)``: probabilities over K classes, summing to 1
        per voxel, or all 0 where the voxel is not scored (outside the core).

    Returns
    -------
    torch.Tensor
        Float32 ``(nx, ny, nz)`` in [0, 1]: 0 where one class is certain or the
        probabilities are all 0, 1 where all K classes are equally likely.
    """
    p = p.float()
    # 0 log 0 = 0: the clamp only keeps log finite, and the product is still 0
    return -(p * p.clamp_min(1e-12).log()).sum(0) / math.log(p.shape[0])


def main(argv: list[str] | None = None):
    """Run the ``cfd-rgba-net-infer`` command line.

    Parameters
    ----------
    argv : list[str] or None, optional
        The arguments. ``None`` means ``sys.argv[1:]``.
    """
    from .build_case import setup_threads
    from .evaluate import predict
    from .geometry import read_body
    from .model import UNet3D
    from .preprocess import read_volume
    from .render import (
        VIEWS,
        load_config,
        render_probabilities,
        render_volumes,
        write_png,
    )

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "volume",
        type=Path,
        help="the case's volume mesh (.vtu), or the net's inputs (.vti)",
    )
    ap.add_argument(
        "--arrays",
        nargs="+",
        help="point arrays of a .vti volume, in the net's channel order",
    )
    ap.add_argument(
        "--method",
        choices=("net", "baseline", "labels"),
        default="net",
        help="what is rendered",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        help="model weights (UNet3D, 5 levels); --method net only",
    )
    ap.add_argument(
        "--tau-lambda2",
        type=float,
        help="vortex threshold the .vtu's inputs are squashed with "
        "(default: the label rules'); --method net only",
    )
    ap.add_argument(
        "--tau-omega",
        type=float,
        help="shear-layer threshold the .vtu's inputs are squashed with "
        "(default: the label rules'); --method net only",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output prefix: <out>_<view>.png",
    )
    ap.add_argument(
        "--body",
        type=Path,
        help="body STL: gives H and the body mask, and is drawn in the renders",
    )
    ap.add_argument(
        "--H",
        type=float,
        help="reference length in mesh units, needed without --body",
    )
    ap.add_argument(
        "--dataset",
        default="ahmedml",
        help="dataset config, a name in configs/datasets/ or a YAML path",
    )
    ap.add_argument(
        "--contract",
        type=Path,
        help="input contract YAML (default: configs/contract_v1.yaml)",
    )
    ap.add_argument(
        "--rules",
        type=Path,
        help="label rules YAML (default: configs/label_rules_v1.yaml)",
    )
    ap.add_argument(
        "--views",
        nargs="+",
        default=["side", "top", "rear"],
        choices=sorted(VIEWS),
    )
    ap.add_argument(
        "--render-config",
        type=Path,
        help="render config YAML (default: configs/render_default.yaml)",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="VTK threads (default: all cores)",
    )
    args = ap.parse_args(argv)
    presquashed = args.volume.suffix == ".vti"
    if presquashed and (args.method != "net" or not args.arrays):
        ap.error("a .vti volume needs --method net and --arrays")
    if not presquashed and args.body is None and args.H is None:
        ap.error("give --body or --H")
    if args.method == "net" and args.checkpoint is None:
        ap.error("--method net needs --checkpoint")
    # VTK's default SMP backend probes the mesh about 4x slower: 325 s against
    # 72 s on an AhmedML case
    setup_threads(args.threads)

    t0 = time.time()

    def step(msg):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        print(
            f"[{time.time() - t0:6.1f}s, peak RSS {rss:.1f} GB] {msg}",
            flush=True,
        )

    cfg, contract, rules = (
        CaseConfig.load(args.dataset),
        Contract.load(args.contract),
        LabelRules.load(args.rules),
    )
    body = (
        read_body(args.body, cfg.body_scale, cfg.body_merge_tolerance)
        if args.body
        else None
    )
    if presquashed:
        q, spec = read_inputs(args.volume, args.arrays)
        core = torch.ones(spec.dims, dtype=torch.bool)
        step(f"read {q.shape[0]} input channels on a {spec.dims} grid")
        if q.abs().max() > 1:
            raise SystemExit(f"{args.volume}: inputs must lie in [-1, 1]")
    else:
        ug = read_volume(args.volume, cfg.field_names())
        step(f"read {args.volume.name}: {ug.GetNumberOfCells():,} cells")
        if args.method == "labels":
            q, spec, core = case_ingredients(
                ug, cfg, contract, rules, body, args.H
            )
            step(
                f"native-mesh ingredients on a {spec.dims} grid, "
                f"dx = {spec.dx:.6g}"
            )
        else:
            q, spec, core = case_features(
                ug, cfg, contract, rules, body, args.H
            )
            step(f"grid features on a {spec.dims} grid, dx = {spec.dx:.6g}")
        del ug

    if args.method == "net":
        if not presquashed:
            # the user's thresholds replace the rules' for the net's inputs
            # only; the core still comes from the rules
            taus = {
                k: v
                for k, v in (
                    ("tau_lambda2", args.tau_lambda2),
                    ("tau_omega", args.tau_omega),
                )
                if v is not None
            }
            squash_rules = dataclasses.replace(rules, **taus)
            q = torch.from_numpy(squash_rules.squash(q.numpy()))
            step(
                f"squashed with tau_lambda2 = {squash_rules.tau_lambda2}, "
                f"tau_omega = {squash_rules.tau_omega}"
            )
        device = default_device()
        state = torch.load(
            args.checkpoint, map_location=device, weights_only=True
        )
        # the checkpoint's first and last convolutions give n_in and n_out
        n_in, n_out = (
            state["enc.0.0.weight"].shape[1],
            state["head.weight"].shape[0],
        )
        if n_in != q.shape[0]:
            raise SystemExit(
                f"{args.checkpoint} takes {n_in} inputs, got {q.shape[0]}"
            )
        if n_out != 3:  # the renders draw none, vortex and shear layer
            raise SystemExit(
                f"{args.checkpoint} has {n_out} classes; rendering needs 3"
            )
        model = UNet3D(n_in, n_out, levels=5).to(device).eval()
        model.load_state_dict(state)
        p = predict(model, q, device).cpu() * core
        step(f"probabilities on {device}")
    else:
        vortex, shear = rules.apply(q, core)
        p = torch.stack([~(vortex | shear), vortex, shear]).float()
        step("rules applied")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.render_config)
    images = render_probabilities(p, spec, config, tuple(args.views), body)
    for name, image in images.items():
        write_png(f"{args.out}_{name}.png", image)
    step(f"rendered {', '.join(args.views)}")

    if args.method == "net":
        u = normalized_entropy(p)
        n_core = int(core.sum())
        frac = {
            t: float((u > t).sum()) / max(n_core, 1) for t in (0.05, 0.3, 0.63)
        }
        images = render_volumes(
            [(u.numpy(), config["uncertainty"])],
            spec,
            config,
            tuple(args.views),
            body,
        )
        for name, image in images.items():
            write_png(f"{args.out}_{name}_uncertainty.png", image)
        step(
            "rendered uncertainty; of the core voxels, "
            f"{frac[0.05]:.3%} above 0.05, {frac[0.3]:.3%} above 0.3 "
            f"and {frac[0.63]:.3%} above 0.63 "
            "(an even split between two classes)"
        )


if __name__ == "__main__":
    main()
