"""Body geometry, the voxel grid (on the mesh's own axes) and the signed distance to the body.

A "voxel" here is a grid point of a ``vtkImageData``: arrays hold one value per grid point, and the
grid's bounds run from the first to the last point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import VTK_FLOAT, VTK_UNSIGNED_CHAR, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkImplicitPolyDataDistance
from vtkmodules.vtkImagingHybrid import vtkSampleFunction
from vtkmodules.vtkImagingStencil import vtkImageStencilToImage, vtkPolyDataToImageStencil
from vtkmodules.vtkIOGeometry import vtkSTLReader

from .config import AXIS_NAMES, Axis, Contract


def read_body(path: str | Path, scale: float = 1.0, merge_tolerance: float | None = None) -> vtkPolyData:
    """Read the body surface from an STL file, optionally cleaning it for distance queries.

    The STL reader itself merges identical vertices. Some datasets (WindsorML) write the same vertex
    with ~1e-6 m differences, which leaves the surface open; for those, ``merge_tolerance`` merges
    vertices that are close but not identical, then drops the triangles that collapse when their points
    merge. They are dropped, not turned into lines or vertices, because ``vtkImplicitPolyDataDistance``
    crashes on those.

    Parameters
    ----------
    path : str or pathlib.Path
        The STL file.
    scale : float, optional
        Factor from STL units to mesh units. Multiplies the point coordinates.
    merge_tolerance : float or None, optional
        Vertices closer than this fraction of the bounding-box diagonal are merged. ``None`` skips
        the cleaning step.

    Returns
    -------
    vtkPolyData
        The cleaned, scaled body surface.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    ValueError
        If the file holds no triangles.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"body STL not found: {path}")
    reader = vtkSTLReader(file_name=str(path))
    if merge_tolerance is None:
        reader.Update()
        surface = reader.GetOutput()
    else:
        clean = vtkCleanPolyData(input_connection=reader.output_port, tolerance=merge_tolerance,
                                 tolerance_is_absolute=False, point_merging=True, convert_polys_to_lines=False,
                                 convert_lines_to_points=False, convert_strips_to_polys=False)
        clean.Update()
        surface = clean.GetOutput()
    body = vtkPolyData()
    body.DeepCopy(surface)
    if body.GetNumberOfCells() == 0:
        raise ValueError(f"body STL {path} holds no triangles")
    if scale != 1.0:
        pts = vtk_to_numpy(body.GetPoints().GetData()).astype(np.float64) * scale
        new = vtkPoints()
        new.SetData(numpy_to_vtk(pts, deep=True))
        body.SetPoints(new)
    return body


@dataclass(frozen=True)
class ImageSpec:
    """A uniform voxel grid on the mesh's own axes. Arrays are indexed ``[i, j, k] = (x, y, z)``.

    Attributes
    ----------
    origin : tuple[float, float, float]
        Coordinates of voxel ``[0, 0, 0]``.
    dx : float
        Voxel size, the same along every axis.
    dims : tuple[int, int, int]
        Number of voxels along x, y and z.
    """
    origin: tuple[float, float, float]
    dx: float
    dims: tuple[int, int, int]

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """The ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the first and last voxel."""
        o, d, n = self.origin, self.dx, self.dims
        return (o[0], o[0] + (n[0] - 1) * d, o[1], o[1] + (n[1] - 1) * d, o[2], o[2] + (n[2] - 1) * d)

    def axis(self, i: int) -> np.ndarray:
        """Compute the voxel coordinates along one axis.

        Parameters
        ----------
        i : int
            The axis: 0, 1 or 2 for x, y or z.

        Returns
        -------
        numpy.ndarray
            Shape ``(dims[i],)``: the coordinate of each voxel along axis ``i``.
        """
        return self.origin[i] + self.dx * np.arange(self.dims[i])

    def to_vtk(self) -> vtkImageData:
        """Build an empty ``vtkImageData`` with this grid.

        Returns
        -------
        vtkImageData
            An image with this spec's dimensions, spacing and origin, and no arrays.
        """
        return vtkImageData(dimensions=self.dims, spacing=(self.dx,) * 3, origin=self.origin)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict that serializes to JSON.

        Returns
        -------
        dict[str, Any]
            ``origin`` and ``dims`` as lists, and ``dx``.
        """
        return dict(origin=list(self.origin), dx=self.dx, dims=list(self.dims))


@dataclass(frozen=True)
class CaseGeometry:
    """Body and ground on the mesh's own axes.

    Attributes
    ----------
    body_bounds : tuple[float, float, float, float, float, float]
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the body.
    ground : float
        Ground coordinate along the up axis.
    H : float
        Reference length, in mesh units.
    up : Axis
        The mesh axis, with sign, that points up away from the ground.
    """
    body_bounds: tuple[float, float, float, float, float, float]
    ground: float
    H: float
    up: Axis

    def height_above_ground(self, spec: ImageSpec) -> np.ndarray:
        """Compute the height of each voxel layer along the up axis.

        Parameters
        ----------
        spec : ImageSpec
            The voxel grid.

        Returns
        -------
        numpy.ndarray
            The height, in mesh units, of each layer along the up axis, positive above the ground. Shaped
            to broadcast against ``(nx, ny, nz)``: size ``dims[up.index]`` on the up axis, 1 elsewhere.
        """
        h = self.up.sign * (spec.axis(self.up.index) - self.ground)
        shape = [1, 1, 1]
        shape[self.up.index] = -1
        return h.reshape(shape)

    def image_spec(self, contract: Contract, domain: tuple[float, float, float, float, float, float],
                   floor_offset_H: float = 0.0, grid_inset_H: float = 0.0) -> ImageSpec:
        """Build the voxel grid that fills a domain.

        Voxels have size ``H / contract.n_per_H``. On each axis the grid starts ``grid_inset_H * H`` inside
        the domain's low face and holds as many voxels as fit before ``grid_inset_H * H`` inside its high
        face. The last voxel therefore falls up to one voxel short of that limit, not on it. The inset
        keeps voxels off the mesh's faces, where the resampling probe misses points at random
        (floating-point ties).

        On the up axis the grid starts ``floor_offset_H * H`` above the ground, in place of the inset,
        when the domain's ground-side face is at or beyond the ground. Otherwise the ground-side face
        gets the ordinary inset. When up is a negative axis, the grid is anchored at the ground side, so
        the first layer sits ``floor_offset_H * H`` below the ground and the far side falls short.

        Parameters
        ----------
        contract : Contract
            The input contract. Supplies ``n_per_H``.
        domain : tuple[float, float, float, float, float, float]
            ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the region to cover.
        floor_offset_H : float, optional
            Distance, in units of H, of the first layer from the ground.
        grid_inset_H : float, optional
            Distance, in units of H, that the grid stays inside the domain's side faces.

        Returns
        -------
        ImageSpec
            The voxel grid.

        Raises
        ------
        ValueError
            If the domain, after the inset and the ground offset, is empty along some axis.
        """
        dx = self.H / contract.n_per_H
        eps = grid_inset_H * self.H
        floor = floor_offset_H * self.H
        lo, n = [0.0] * 3, [0] * 3
        for a in range(3):
            start, end = domain[2 * a] + eps, domain[2 * a + 1] - eps
            if a == self.up.index:
                if self.up.sign > 0 and domain[2 * a] <= self.ground:
                    start = self.ground + floor
                elif self.up.sign < 0 and domain[2 * a + 1] >= self.ground:
                    end = self.ground - floor
            if not end > start:
                raise ValueError(f"empty domain along {AXIS_NAMES[a]}: {domain}, ground {self.ground}")
            # the 1e-9 keeps a span that is an exact multiple of dx from losing its last voxel to round-off
            n[a] = int(np.floor((end - start) / dx + 1e-9)) + 1
            # anchor at the ground side along the up axis, else at the low face
            lo[a] = end - (n[a] - 1) * dx if (a == self.up.index and self.up.sign < 0) else start
        return ImageSpec(tuple(float(v) for v in lo), dx, tuple(n))  # type: ignore[arg-type]


def case_geometry(body: vtkPolyData, ground: float, reference_length: float | None, up: Axis) -> CaseGeometry:
    """Derive the case geometry from the body surface.

    Parameters
    ----------
    body : vtkPolyData
        The body surface, in mesh units.
    ground : float
        Ground coordinate along the up axis.
    reference_length : float or None
        The reference length H. ``None`` means the body's extent along the up axis.
    up : Axis
        The mesh axis, with sign, that points up away from the ground.

    Returns
    -------
    CaseGeometry
        The body bounds, ground, H and up axis.

    Raises
    ------
    ValueError
        If H is not positive, or the body reaches below the ground.
    """
    b = body.GetBounds()
    j = up.index
    # extent along up, measured so that larger means higher whichever way the axis points
    lowest, highest = (b[2 * j], b[2 * j + 1]) if up.sign > 0 else (-b[2 * j + 1], -b[2 * j])
    H = reference_length if reference_length is not None else highest - lowest
    if not H > 0:
        raise ValueError(f"reference length H must be > 0, got {H}")
    # a body that touches the ground (wheels) is fine; the 1e-6 H slack absorbs round-off
    if lowest < up.sign * ground - 1e-6 * H:
        raise ValueError(f"body reaches below the ground along {up}: body {b}, ground {ground}")
    return CaseGeometry(tuple(b), float(ground), float(H), up)  # type: ignore[arg-type]


def body_inside(body: vtkPolyData, spec: ImageSpec) -> np.ndarray:
    """Find the voxels inside the closed body surface.

    Uses scanline parity, so the result does not depend on the surface's normals.

    Parameters
    ----------
    body : vtkPolyData
        The closed body surface.
    spec : ImageSpec
        The voxel grid.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``spec.dims``, true for voxels inside the body.
    """
    n = spec.dims
    stencil = vtkPolyDataToImageStencil(input_data=body, output_origin=spec.origin, output_spacing=(spec.dx,) * 3,
                                        output_whole_extent=(0, n[0] - 1, 0, n[1] - 1, 0, n[2] - 1))
    to_image = vtkImageStencilToImage(input_connection=stencil.output_port, inside_value=1, outside_value=0,
                                      output_scalar_type=VTK_UNSIGNED_CHAR)
    to_image.Update()
    return vtk_to_numpy(to_image.GetOutput().GetPointData().GetScalars()).reshape(n, order="F").astype(bool)


def _exact_distance(body: vtkPolyData, spec: ImageSpec) -> np.ndarray:
    """Compute the unsigned distance from every voxel to the body's triangles.

    Uses ``vtkImplicitPolyDataDistance``, which is slow on large grids.

    Parameters
    ----------
    body : vtkPolyData
        The body surface.
    spec : ImageSpec
        The voxel grid.

    Returns
    -------
    numpy.ndarray
        Array of shape ``spec.dims``: the distance, in mesh units, from each voxel to the surface.
    """
    dist = vtkImplicitPolyDataDistance()
    dist.SetInput(body)
    sampler = vtkSampleFunction(implicit_function=dist, model_bounds=spec.bounds, sample_dimensions=spec.dims,
                                compute_normals=False, output_scalar_type=VTK_FLOAT)
    sampler.Update()
    return np.abs(vtk_to_numpy(sampler.GetOutput().GetPointData().GetScalars())).reshape(spec.dims, order="F")


def signed_distance(body: vtkPolyData, spec: ImageSpec, band: float | None = None,
                    info: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute the signed distance from each voxel to the body surface.

    The sign comes from ``body_inside`` (scanline parity), because the implicit function's normal-based
    sign is wrong at some far-field points near sharp edges. Training and inference both call this, so
    they get identical values.

    With ``band=None`` the distance is exact everywhere (``vtkImplicitPolyDataDistance``), which is slow
    on large grids. With a ``band``, it is exact only inside the body's bounding box grown by ``band``.
    Outside that box it is the Euclidean distance transform of the inside mask (distance to the nearest
    inside voxel centre), shifted by one offset so it meets the exact values on the box's boundary. Its
    error is a fraction of a voxel, small against distances greater than ``band``.

    Parameters
    ----------
    body : vtkPolyData
        The closed body surface.
    spec : ImageSpec
        The voxel grid.
    band : float or None, optional
        Half-width, in mesh units, of the margin around the body's bounding box where the distance is
        exact. ``None`` means exact everywhere.
    info : dict or None, optional
        Output argument. When given, receives ``method`` and, with a band, ``band`` and ``edt_offset``
        (the shift applied to the distance transform).

    Returns
    -------
    sdf : numpy.ndarray
        Float32 array of shape ``spec.dims``: the signed distance in mesh units, negative inside.
    inside : numpy.ndarray
        Boolean array of shape ``spec.dims``, true for voxels inside the body.

    Raises
    ------
    ValueError
        If no voxel lies inside the body.
    """
    inside = body_inside(body, spec)
    if not inside.any():
        raise ValueError(f"no voxel lies inside the body (body bounds {tuple(round(b, 6) for b in body.GetBounds())}, "
                         f"grid bounds {tuple(round(b, 6) for b in spec.bounds)}); check `domain` and `body.scale` "
                         "in the dataset config")
    if band is None:
        d = _exact_distance(body, spec)
        if info is not None:
            info.update(method="exact")
        return np.where(inside, -d, d).astype(np.float32), inside

    # index ranges of the band box, clipped to the grid
    b = body.GetBounds()
    sl = []
    for a in range(3):
        ax = spec.axis(a)
        i0 = int(np.searchsorted(ax, b[2 * a] - band, side="left"))
        i1 = int(np.searchsorted(ax, b[2 * a + 1] + band, side="right"))
        sl.append(slice(max(i0, 0), min(i1, spec.dims[a])))
    sub = ImageSpec(tuple(spec.origin[a] + spec.dx * sl[a].start for a in range(3)), spec.dx,  # type: ignore[arg-type]
                    tuple(sl[a].stop - sl[a].start for a in range(3)))
    exact = _exact_distance(body, sub)

    d = distance_transform_edt(~inside, sampling=spec.dx).astype(np.float32)
    # offset that makes the transform meet the exact distance on the band box's outer layer
    in_band = np.zeros(spec.dims, bool)
    in_band[tuple(sl)] = True
    rim = in_band & ~binary_erosion(in_band, border_value=0) & ~inside
    rim_sub = rim[tuple(sl)]
    offset = float(np.median(exact[rim_sub] - d[tuple(sl)][rim_sub])) if rim_sub.any() else 0.0
    d += np.float32(offset)
    d[tuple(sl)] = exact
    if info is not None:
        info.update(method="exact in band, distance transform outside", band=band, edt_offset=offset)
    return np.where(inside, -d, d).astype(np.float32), inside
