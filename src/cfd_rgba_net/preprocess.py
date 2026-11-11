"""
Native mesh -> model inputs. Shared by training-data generation and inference, so the model always
sees inputs built the same way.

    mesh  = read_volume(path, case_cfg.field_names())
    prep  = prepare_mesh(mesh, case_cfg)              # select fields, non-dimensionalize, find the ground
    body  = read_body(stl, case_cfg.body_scale, case_cfg.body_merge_tolerance)
    geo   = case_geometry(body, prep.ground, case_cfg.reference_length, axis_of(case_cfg.flow.up_direction))
    spec  = geo.image_spec(contract, case_cfg.domain or mesh.GetBounds(),
                           case_cfg.floor_offset_H, case_cfg.grid_inset_H)   # the whole domain
    image = resample(prep.mesh, spec)                 # point arrays on the voxel grid
    inp   = assemble_inputs(image, body, geo, spec, contract)   # (6, nx, ny, nz), INPUT_CHANNELS

Nothing is rotated: the voxel grid and the u, v, w channels use the mesh's own axes, whichever way
the flow runs along them. The grid fills the dataset's domain; the up direction only sets H, the
ground, the ground proximity and the floor exclusion.

Body and ground enter as two proximity channels, not as 0/1 masks and not as the raw signed distance
(the ranges below are the contract's body_proximity_range_H and ground_proximity_range_H in v1.0):
    body_proximity   b = max(0, 1 - sdf / (4 H))   1 on the body surface, 0 at 4 H and beyond; inside the body
                                                   the line continues (sdf < 0), so b > 1 inside, b < 1 outside
    ground_proximity g = max(0, 1 - h / H)          1 on the ground, 0 at 1 H above it and higher
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import (vtkGenericCell, vtkPolyData, vtkStaticCellLocator,
                                           vtkUnstructuredGrid)
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData, vtkResampleToImage
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

from .config import FIELD_SOURCES, CaseConfig, Contract, VectorField, axis_of
from .geometry import CaseGeometry, ImageSpec, signed_distance

# Array names on the prepared mesh.
U_ND = "U_nd"      # point data, velocity / U_inf, mesh axes
CP = "Cp"          # point data
U_CELL = "U_cell"  # cell data, velocity / U_inf (gradients only); present when the mesh has cell velocity

# Input channels, in this order (configs/contract_v1.yaml `channels` must match). Changing them, or the
# proximity ranges and the SDF band in the contract, changes the numbers a model sees: bump CONTRACT_VERSION.
INPUT_CHANNELS = ("u", "v", "w", "Cp", "body_proximity", "ground_proximity")

# vtkGenericCell.GetPointId / FindCell weights: the most points a cell may have in probe_point.
MAX_CELL_POINTS = 64


def body_proximity(sdf_H: np.ndarray, range_H: float) -> np.ndarray:
    """Compute the body proximity channel from the signed distance.

    Parameters
    ----------
    sdf_H : numpy.ndarray
        Signed distance to the body divided by H, negative inside the body.
    range_H : float
        Distance, in units of H, at which the channel reaches 0 (``Contract.body_proximity_range_H``).

    Returns
    -------
    numpy.ndarray
        Float32 array, same shape as ``sdf_H``: ``max(0, 1 - sdf_H / range_H)``. It is 1 on the surface,
        above 1 inside the body, and 0 from ``range_H`` out.
    """
    return np.maximum(0.0, 1.0 - np.asarray(sdf_H, np.float32) / np.float32(range_H))


def ground_proximity(h_H: np.ndarray, range_H: float) -> np.ndarray:
    """Compute the ground proximity channel from the height above the ground.

    Parameters
    ----------
    h_H : numpy.ndarray
        Height above the ground divided by H.
    range_H : float
        Height, in units of H, at which the channel reaches 0 (``Contract.ground_proximity_range_H``).

    Returns
    -------
    numpy.ndarray
        Float32 array, same shape as ``h_H``: ``max(0, 1 - h_H / range_H)``. It is 1 on the ground and
        0 from ``range_H`` up.
    """
    return np.maximum(0.0, 1.0 - np.asarray(h_H, np.float32) / np.float32(range_H))


def read_volume(path: str | Path, arrays: set[str] | None = None) -> vtkUnstructuredGrid:
    """Read a volume mesh from a ``.vtu`` file.

    With ``arrays``, the reader loads only those point and cell arrays: ``UpdateInformation`` reads just
    the XML header, which fills the reader's array selections, and the others are switched off before
    ``Update``, so their data is never loaded.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.vtu`` file.
    arrays : set[str] or None, optional
        Names of the point and cell arrays to load. ``None`` loads all of them.

    Returns
    -------
    vtkUnstructuredGrid
        The mesh.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    ValueError
        If the mesh holds no points.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"volume mesh not found: {path}")
    reader = vtkXMLUnstructuredGridReader(file_name=str(path))
    if arrays is not None:
        reader.UpdateInformation()
        for sel in (reader.GetPointDataArraySelection(), reader.GetCellDataArraySelection()):
            for name in [sel.GetArrayName(i) for i in range(sel.GetNumberOfArrays())]:
                sel.SetArraySetting(name, int(name in arrays))
    reader.Update()
    ug = reader.GetOutput()
    if ug.GetNumberOfPoints() == 0:
        raise ValueError(f"volume mesh {path} holds no points")
    return ug


@dataclass
class PreparedMesh:
    """The mesh with the model's fields, and the reference values they were scaled with.

    Attributes
    ----------
    mesh : vtkUnstructuredGrid
        The prepared mesh, see ``prepare_mesh``.
    U_inf : float
        Freestream speed. From the config, or read at ``flow.reference_probe``.
    p_inf : float
        Freestream pressure. From the config, or read at ``flow.reference_probe``.
    ground : float
        Ground position: a coordinate along the up axis.
    velocity_source : str
        Where the velocity came from, for logs.
    pressure_source : str
        Where the pressure came from, for logs.
    """
    mesh: vtkUnstructuredGrid
    U_inf: float
    p_inf: float
    ground: float
    velocity_source: str
    pressure_source: str


def _find_vector(ug: vtkUnstructuredGrid, specs: Iterable[VectorField], assoc: str) -> tuple[str | None, np.ndarray | None]:
    """Find the first configured velocity field in the mesh.

    Parameters
    ----------
    ug : vtkUnstructuredGrid
        The mesh.
    specs : iterable of VectorField
        Candidate fields, in order of preference. A named field must have three components.
    assoc : str
        ``"point"`` or ``"cell"``: which data to look in.

    Returns
    -------
    name : str or None
        The field's label, or ``None`` if no candidate is present.
    array : numpy.ndarray or None
        Shape ``(N, 3)``, or ``None`` if no candidate is present.
    """
    data = ug.GetPointData() if assoc == "point" else ug.GetCellData()
    for s in specs:
        if s.name is not None and data.HasArray(s.name):
            a = vtk_to_numpy(data.GetArray(s.name))
            if a.ndim == 2 and a.shape[1] == 3:
                return s.name, a
        elif s.components is not None and all(data.HasArray(c) for c in s.components):
            return "+".join(s.components), np.stack([vtk_to_numpy(data.GetArray(c)) for c in s.components], 1)
    return None, None


def _find_scalar(ug: vtkUnstructuredGrid, names: Iterable[str], assoc: str) -> tuple[str | None, np.ndarray | None]:
    """Find the first configured scalar array in the mesh.

    Parameters
    ----------
    ug : vtkUnstructuredGrid
        The mesh.
    names : iterable of str
        Candidate array names, in order of preference.
    assoc : str
        ``"point"`` or ``"cell"``: which data to look in.

    Returns
    -------
    name : str or None
        The array's name, or ``None`` if no candidate is present.
    array : numpy.ndarray or None
        Shape ``(N,)``, or ``None`` if no candidate is present.
    """
    data = ug.GetPointData() if assoc == "point" else ug.GetCellData()
    for n in names:
        if data.HasArray(n):
            return n, vtk_to_numpy(data.GetArray(n))
    return None, None


def _vtk(name: str, a: np.ndarray):
    """Wrap a NumPy array as a named VTK array, copying the data.

    Parameters
    ----------
    name : str
        The array's name.
    a : numpy.ndarray
        The values, shape ``(N,)`` or ``(N, C)``.

    Returns
    -------
    vtkDataArray
        A copy of ``a`` named ``name``.
    """
    v = numpy_to_vtk(np.ascontiguousarray(a), deep=True)
    v.SetName(name)
    return v


def prepare_mesh(ug: vtkUnstructuredGrid, cfg: CaseConfig, keep_checks: tuple[str, ...] = ()) -> PreparedMesh:
    """Select the velocity and pressure, non-dimensionalize them and find the ground.

    Velocity and pressure come from the data that ``cfg.field_source`` names, with no fallback to the
    other kind. For ``"cell"``, they are averaged to the points with ``vtkCellDataToPointData``. Whatever
    the source, the cell velocity is also kept as ``U_cell`` when the mesh has it, because the label
    ingredients take their gradients of it. Nothing is rotated.

    ``U_inf`` and ``p_inf`` come from the config, or from the mesh at ``flow.reference_probe``. The
    velocity becomes ``U_nd = U / U_inf`` and the pressure ``Cp = 2 (p - p_inf) / (rho U_inf^2)``.

    Parameters
    ----------
    ug : vtkUnstructuredGrid
        The native mesh. Left unchanged.
    cfg : CaseConfig
        The dataset configuration.
    keep_checks : tuple[str, ...], optional
        Names of dataset arrays to carry over as found, for sanity checks.

    Returns
    -------
    PreparedMesh
        A new grid that shares ``ug``'s points and cells, with only:

        - point data: ``U_nd`` (3), ``Cp``, and the arrays named in ``keep_checks``;
        - cell data: ``U_cell`` (3) when the mesh has cell velocity, and the arrays named in
          ``keep_checks``;

        together with ``U_inf``, ``p_inf``, the ground coordinate and the field sources.

    Raises
    ------
    KeyError
        If the mesh's ``field_source`` data holds no configured velocity or no configured pressure.
    ValueError
        If ``cfg.field_source`` is not ``"point"`` or ``"cell"``; if ``flow.rho`` or ``U_inf`` is not
        positive; or if the reference probe lies outside the mesh.
    """
    f = cfg.flow
    source = cfg.field_source
    if source not in FIELD_SOURCES:
        raise ValueError(f"field_source must be one of {FIELD_SOURCES}, got {source!r}")
    if not f.rho > 0:
        raise ValueError(f"flow.rho must be > 0, got {f.rho}")

    u_name, U = _find_vector(ug, cfg.velocity, source)
    p_name, p = _find_scalar(ug, cfg.pressure, source)
    missing = []
    if U is None:
        missing.append(f"velocity {[v.label() for v in cfg.velocity]}")
    if p is None:
        missing.append(f"pressure {list(cfg.pressure)}")
    if missing:
        raise KeyError(f"field_source is '{source}', but the mesh's {source} data has no " + " and no ".join(missing))

    if source == "cell":
        Uc = U
        tmp = vtkUnstructuredGrid()
        tmp.ShallowCopy(ug)
        tmp.GetPointData().Initialize()
        tmp.GetCellData().Initialize()
        tmp.GetCellData().AddArray(_vtk("U", U))
        tmp.GetCellData().AddArray(_vtk("p", p))
        c2p = vtkCellDataToPointData(input_data=tmp)
        c2p.Update()
        U = vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("U"))
        p = vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("p"))
        u_name, p_name = f"cell:{u_name} -> points", f"cell:{p_name} -> points"
    else:
        _, Uc = _find_vector(ug, cfg.velocity, "cell")    # None when the mesh has no cell velocity

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
        U_inf_origin = f"|U| at flow.reference_probe {tuple(f.reference_probe)}"
    else:
        U_inf, p_inf = float(f.U_inf), float(f.p_inf)  # type: ignore[arg-type]
        U_inf_origin = "flow.U_inf"
    if not U_inf > 0:
        raise ValueError(f"U_inf must be > 0, got {U_inf:g} ({U_inf_origin})")
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
        up = axis_of(f.up_direction, "up direction")
        b = out.GetBounds()
        ground = float(b[2 * up.index] if up.sign > 0 else b[2 * up.index + 1])
    return PreparedMesh(out, U_inf, p_inf, ground, u_name, p_name)


def probe_point(ug: vtkUnstructuredGrid, x: Sequence[float], U: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float]:
    """Interpolate the point velocity and pressure at one coordinate.

    Parameters
    ----------
    ug : vtkUnstructuredGrid
        The mesh. Its cells must have at most ``MAX_CELL_POINTS`` points.
    x : sequence of float
        The coordinate ``(x, y, z)``, in mesh units.
    U : numpy.ndarray
        Point velocity of ``ug``, shape ``(N, 3)``.
    p : numpy.ndarray
        Point pressure of ``ug``, shape ``(N,)``.

    Returns
    -------
    U_x : numpy.ndarray
        Velocity at ``x``, shape ``(3,)``.
    p_x : float
        Pressure at ``x``.

    Raises
    ------
    ValueError
        If ``x`` is not inside any cell of ``ug``.
    """
    loc = vtkStaticCellLocator(data_set=ug)
    loc.BuildLocator()
    cell, pcoords, weights = vtkGenericCell(), [0.0] * 3, [0.0] * MAX_CELL_POINTS
    cid = loc.FindCell(list(map(float, x)), 1e-12, cell, pcoords, weights)
    if cid < 0:
        raise ValueError(f"flow.reference_probe {tuple(x)} is not inside any mesh cell")
    ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
    w = np.array(weights[:len(ids)])
    return (w[:, None] * U[ids]).sum(0), float((w * p[ids]).sum())


def resample(mesh: vtkUnstructuredGrid, spec: ImageSpec, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Probe the mesh's point arrays at the voxels of a grid.

    Parameters
    ----------
    mesh : vtkUnstructuredGrid
        The mesh whose point arrays are sampled.
    spec : ImageSpec
        The voxel grid.
    names : list[str] or None, optional
        The point arrays to sample. ``None`` samples all of them.

    Returns
    -------
    dict[str, numpy.ndarray]
        Each sampled array by name, shaped ``(nx, ny, nz)`` for a scalar or ``(C, nx, ny, nz)`` for
        a vector, indexed ``[i, j, k] = (x, y, z)``. The extra entry ``"valid"`` is a boolean
        ``(nx, ny, nz)`` array, true where a mesh cell holds the voxel.
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
    # the mask is a char array, which NumPy can return as bytes ('S1'): read the same bytes as integers
    if valid.dtype.kind == "S":
        valid = valid.view(np.uint8)
    out["valid"] = valid.reshape(spec.dims, order="F") > 0
    return out


@dataclass
class Inputs:
    """The model's input array, the fluid mask and the sanity checks of one case.

    Attributes
    ----------
    inputs : numpy.ndarray
        Float32, shape ``(6, nx, ny, nz)``, channels in ``INPUT_CHANNELS`` order.
    fluid : numpy.ndarray
        Boolean ``(nx, ny, nz)``: inside the mesh and outside the body. Used for labels only; not stored.
    checks : dict
        Voxel counts and other diagnostics, for the case metadata.
    """
    inputs: np.ndarray
    fluid: np.ndarray
    checks: dict


def assemble_inputs(image: dict[str, np.ndarray], body: vtkPolyData, geo: CaseGeometry, spec: ImageSpec,
                    contract: Contract) -> Inputs:
    """Build the model's input channels on the voxel grid.

    Computes the signed distance to the body, marks the fluid voxels (inside the mesh, outside the body),
    zeroes ``u, v, w, Cp`` elsewhere, and adds the body and ground proximity channels.

    Parameters
    ----------
    image : dict[str, numpy.ndarray]
        The resampled prepared mesh from ``resample``: needs ``U_nd``, ``Cp`` and ``valid``.
    body : vtkPolyData
        The closed body surface.
    geo : CaseGeometry
        Body and ground geometry.
    spec : ImageSpec
        The voxel grid that ``image`` was resampled on.
    contract : Contract
        The input contract. Supplies the proximity ranges and the signed-distance band.

    Returns
    -------
    Inputs
        The stacked channels, the fluid mask and the checks.

    Raises
    ------
    ValueError
        If no voxel lies inside the body.
    """
    sdf_info: dict = {}
    sdf, inside = signed_distance(body, spec, band=contract.sdf_exact_band_H * geo.H, info=sdf_info)
    valid = image["valid"]
    fluid = valid & ~inside
    U = image[U_ND].astype(np.float32)
    Cp = image[CP].astype(np.float32)
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
    sdf_H = (sdf / geo.H).astype(np.float32)
    h_H = np.broadcast_to(geo.height_above_ground(spec) / geo.H, spec.dims)
    b = body_proximity(sdf_H, contract.body_proximity_range_H)
    g = ground_proximity(h_H, contract.ground_proximity_range_H)
    checks["proximity"] = dict(
        body_range_H=contract.body_proximity_range_H, ground_range_H=contract.ground_proximity_range_H,
        body_max=float(b.max()),                              # 1 + max depth inside the body / body range
        n_body_gt1_not_body=int(((b > 1) & ~inside).sum()),   # 0: b > 1 exactly on the body voxels
        ground_first_layer=float(g.max()),                    # 1 - floor_offset_H / range, when the grid starts at the floor
        n_ground_gt0=int((g > 0).sum()),
    )
    inputs = np.stack([U[0], U[1], U[2], Cp, b, g]).astype(np.float32)
    return Inputs(inputs, fluid, checks)
