"""
Native mesh -> model inputs. Shared by training-data generation and inference, so the model always
sees inputs built the same way.

    mesh  = read_volume(path, case_cfg.field_names())
    prep  = prepare_mesh(mesh, case_cfg)              # select fields, non-dimensionalize, find the ground
    body  = read_body(stl, case_cfg.body_scale)
    geo   = case_geometry(body, prep.ground, case_cfg.reference_length, flow.up_axis)
    spec  = geo.image_spec(contract, case_cfg.domain or mesh.GetBounds())   # the whole domain
    image = resample(prep.mesh, spec)                 # point arrays on the voxel grid
    inp   = assemble_inputs(image, body, geo, spec)   # (5, nx, ny, nz) + fluid mask

Nothing is rotated: the voxel grid and the u, v, w channels use the mesh's own axes, whichever way
the flow runs along them. The grid fills the dataset's domain; the up direction only sets H, the
ground and the floor exclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkUnstructuredGrid
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData, vtkResampleToImage
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

from .config import CaseConfig
from .geometry import CaseGeometry, ImageSpec, signed_distance

# Array names on the prepared mesh.
U_ND = "U_nd"      # point data, velocity / U_inf, mesh axes
CP = "Cp"          # point data
U_CELL = "U_cell"  # cell data, velocity / U_inf (gradients only); may be absent

# The SDF is exact within this distance (units of H) of the body's bounding box, a distance transform beyond.
SDF_EXACT_BAND_H = 1.0


def read_volume(path: str | Path, arrays: set[str] | None = None) -> vtkUnstructuredGrid:
    """
    Read a .vtu. With `arrays`, only those point / cell arrays are read: UpdateInformation reads
    just the XML header, which fills the reader's array selections; the others are then switched off
    before Update, so their data is never loaded.
    """
    reader = vtkXMLUnstructuredGridReader(file_name=str(path))
    if arrays is not None:
        reader.UpdateInformation()
        for sel in (reader.GetPointDataArraySelection(), reader.GetCellDataArraySelection()):
            for name in [sel.GetArrayName(i) for i in range(sel.GetNumberOfArrays())]:
                sel.SetArraySetting(name, int(name in arrays))
    reader.Update()
    return reader.GetOutput()


@dataclass
class PreparedMesh:
    mesh: vtkUnstructuredGrid
    U_inf: float                # from the config, or read at flow.reference_probe
    p_inf: float
    ground: float               # ground position: a coordinate along the up axis
    velocity_source: str
    pressure_source: str


def _find_vector(ug: vtkUnstructuredGrid, specs, assoc: str):
    data = ug.GetPointData() if assoc == "point" else ug.GetCellData()
    for s in specs:
        if s.name is not None and data.HasArray(s.name):
            a = vtk_to_numpy(data.GetArray(s.name))
            if a.ndim == 2 and a.shape[1] == 3:
                return s.name, a
        elif s.components is not None and all(data.HasArray(c) for c in s.components):
            return "+".join(s.components), np.stack([vtk_to_numpy(data.GetArray(c)) for c in s.components], 1)
    return None, None


def _find_scalar(ug: vtkUnstructuredGrid, names, assoc: str):
    data = ug.GetPointData() if assoc == "point" else ug.GetCellData()
    for n in names:
        if data.HasArray(n):
            return n, vtk_to_numpy(data.GetArray(n))
    return None, None


def _vtk(name: str, a: np.ndarray):
    v = numpy_to_vtk(np.ascontiguousarray(a), deep=True)
    v.SetName(name)
    return v


def prepare_mesh(ug: vtkUnstructuredGrid, cfg: CaseConfig, keep_checks: tuple[str, ...] = ()) -> PreparedMesh:
    """
    Returns a new grid sharing ug's points and cells with only:
        point data: U_nd (3), Cp, and the arrays named in keep_checks (as found)
        cell data : U_cell (3) when the input has cell velocity, and the arrays named in keep_checks
    Velocity and pressure are taken from point data when present, else converted from cell data with
    vtkCellDataToPointData. Nothing is rotated.
    """
    f = cfg.flow
    u_name, U = _find_vector(ug, cfg.velocity, "point")
    p_name, p = _find_scalar(ug, cfg.pressure, "point")
    uc_name, Uc = _find_vector(ug, cfg.velocity, "cell")
    if U is None or p is None:
        pc_name, pc = _find_scalar(ug, cfg.pressure, "cell")
        if Uc is None or pc is None:
            raise KeyError(f"velocity {[v.label() for v in cfg.velocity]} or pressure {list(cfg.pressure)} "
                           "not found in point or cell data")
        tmp = vtkUnstructuredGrid()
        tmp.ShallowCopy(ug)
        tmp.GetPointData().Initialize()
        tmp.GetCellData().Initialize()
        tmp.GetCellData().AddArray(_vtk("U", Uc))
        tmp.GetCellData().AddArray(_vtk("p", pc))
        c2p = vtkCellDataToPointData(input_data=tmp)
        c2p.Update()
        U = vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("U"))
        p = vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("p"))
        u_name, p_name = f"cell:{uc_name} -> points", f"cell:{pc_name} -> points"

    out = vtkUnstructuredGrid()
    out.ShallowCopy(ug)
    pd, cd = out.GetPointData(), out.GetCellData()
    pd.Initialize()
    cd.Initialize()
    for name in keep_checks:
        for src, dst in ((ug.GetPointData(), pd), (ug.GetCellData(), cd)):
            if src.HasArray(name):
                dst.AddArray(src.GetArray(name))

    if f.reference_probe is not None:
        U_ref, p_ref = probe_point(ug, f.reference_probe, U, p)
        U_inf, p_inf = float(np.linalg.norm(U_ref)), float(p_ref)
    else:
        U_inf, p_inf = float(f.U_inf), float(f.p_inf)  # type: ignore[arg-type]
    U_nd = (U.astype(np.float32) / np.float32(U_inf))
    Cp = (2.0 * (p.astype(np.float64) - p_inf) / (f.rho * U_inf**2)).astype(np.float32)
    pd.AddArray(_vtk(U_ND, U_nd))
    pd.AddArray(_vtk(CP, Cp))
    if Uc is not None:
        Uc_nd = Uc.astype(np.float32) / np.float32(U_inf)
        cd.AddArray(_vtk(U_CELL, Uc_nd))

    if f.ground is not None:
        ground = f.ground
    else:  # the lowest mesh point along the up direction
        up = f.up_axis
        b = out.GetBounds()
        ground = float(b[2 * up.index] if up.sign > 0 else b[2 * up.index + 1])
    return PreparedMesh(out, U_inf, p_inf, ground, u_name, p_name)


def probe_point(ug: vtkUnstructuredGrid, x, U: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float]:
    """Point velocity U (N, 3) and pressure p (N,) of ug, interpolated at x."""
    from vtkmodules.vtkCommonDataModel import vtkGenericCell, vtkStaticCellLocator
    loc = vtkStaticCellLocator(data_set=ug)
    loc.BuildLocator()
    cell, pcoords, weights = vtkGenericCell(), [0.0] * 3, [0.0] * 64
    cid = loc.FindCell(list(map(float, x)), 1e-12, cell, pcoords, weights)
    if cid < 0:
        raise ValueError(f"flow.reference_probe {tuple(x)} is not inside any mesh cell")
    ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
    w = np.array(weights[:len(ids)])
    return (w[:, None] * U[ids]).sum(0), float((w * p[ids]).sum())


def resample(mesh: vtkUnstructuredGrid, spec: ImageSpec, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """
    Probe the mesh's point arrays at the voxel centers. Returns {name: (nx, ny, nz) or (C, nx, ny, nz)}
    plus "valid": True where a mesh cell holds the voxel.
    """
    resampler = vtkResampleToImage(input_data_object=mesh, sampling_dimensions=spec.dims,
                                   sampling_bounds=spec.bounds, use_input_bounds=False)
    resampler.Update()
    pd = resampler.GetOutput().GetPointData()
    out: dict[str, np.ndarray] = {}
    for name in names or [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]:
        a = vtk_to_numpy(pd.GetArray(name))
        if name == "vtkValidPointMask":
            continue
        out[name] = (a.reshape(spec.dims, order="F") if a.ndim == 1
                     else np.stack([a[:, c].reshape(spec.dims, order="F") for c in range(a.shape[1])]))
    valid = vtk_to_numpy(pd.GetArray("vtkValidPointMask"))
    if valid.dtype.kind == "S":
        valid = valid.view(np.uint8)
    out["valid"] = valid.reshape(spec.dims, order="F") > 0
    return out


@dataclass
class Inputs:
    inputs: np.ndarray       # (5, nx, ny, nz) float32: u, v, w, Cp, sdf/H
    fluid: np.ndarray        # (nx, ny, nz) bool: inside the mesh and outside the body
    checks: dict


def assemble_inputs(image: dict[str, np.ndarray], body: vtkPolyData, geo: CaseGeometry, spec: ImageSpec) -> Inputs:
    sdf_info: dict = {}
    sdf, inside = signed_distance(body, spec, band=SDF_EXACT_BAND_H * geo.H, info=sdf_info)
    valid = image["valid"]
    fluid = valid & ~inside
    U = image[U_ND].astype(np.float32).copy()
    Cp = image[CP].astype(np.float32).copy()
    U[:, ~fluid] = 0.0
    Cp[~fluid] = 0.0
    # voxels that are neither in the mesh nor in the body: the box leaves the CFD domain, or the
    # mesh has holes. Either way the inputs there are unreliable, so count them.
    outside_both = ~valid & ~inside
    checks = dict(
        n_voxels=int(valid.size),
        n_fluid=int(fluid.sum()),
        n_body=int(inside.sum()),
        n_not_in_mesh_nor_body=int(outside_both.sum()),
        n_in_mesh_and_body=int((valid & inside).sum()),  # > 0 for immersed-boundary meshes
        Cp_max_fluid=float(Cp[fluid].max()) if fluid.any() else None,  # ~1 at the stagnation point
        sdf=sdf_info,
    )
    inputs = np.concatenate([U, Cp[None], (sdf / geo.H)[None]], axis=0).astype(np.float32)
    return Inputs(inputs, fluid, checks)
