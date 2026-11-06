"""Body geometry, the voxel grid (on the mesh's own axes) and the signed distance to the body."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkImplicitPolyDataDistance
from vtkmodules.vtkImagingHybrid import vtkSampleFunction
from vtkmodules.vtkImagingStencil import vtkImageStencilToImage, vtkPolyDataToImageStencil
from vtkmodules.vtkIOGeometry import vtkSTLReader

from .config import Axis, Contract


# STL vertices closer than this fraction of the bounding-box diagonal are merged. Some datasets
# (WindsorML) write the same vertex with ~1e-6 m differences, which leaves the surface open.
STL_MERGE_TOLERANCE = 1e-5


def read_body(path: str | Path, scale: float = 1.0) -> vtkPolyData:
    reader = vtkSTLReader(file_name=str(path))
    # Triangles that collapse when their points merge are dropped, not turned into lines or vertices:
    # vtkImplicitPolyDataDistance crashes on those.
    clean = vtkCleanPolyData(input_connection=reader.output_port, tolerance=STL_MERGE_TOLERANCE,
                             tolerance_is_absolute=False, point_merging=True, convert_polys_to_lines=False,
                             convert_lines_to_points=False, convert_strips_to_polys=False)
    clean.Update()
    body = vtkPolyData()
    body.DeepCopy(clean.GetOutput())
    if scale != 1.0:
        pts = vtk_to_numpy(body.GetPoints().GetData()).astype(np.float64) * scale
        new = vtkPoints()
        new.SetData(numpy_to_vtk(pts, deep=True))
        body.SetPoints(new)
    return body


@dataclass(frozen=True)
class ImageSpec:
    """A uniform voxel grid on the mesh's own axes. Arrays are indexed [i, j, k] = (x, y, z)."""
    origin: tuple[float, float, float]
    dx: float
    dims: tuple[int, int, int]

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        o, d, n = self.origin, self.dx, self.dims
        return (o[0], o[0] + (n[0] - 1) * d, o[1], o[1] + (n[1] - 1) * d, o[2], o[2] + (n[2] - 1) * d)

    def axis(self, i: int) -> np.ndarray:
        return self.origin[i] + self.dx * np.arange(self.dims[i])

    def to_vtk(self) -> vtkImageData:
        return vtkImageData(dimensions=self.dims, spacing=(self.dx,) * 3, origin=self.origin)

    def to_dict(self):
        return dict(origin=list(self.origin), dx=self.dx, dims=list(self.dims))


# Voxels are kept this far (in units of H) inside the domain's side faces: points exactly on a face of the
# mesh are missed by the probe at random (floating-point ties). The ground side uses the dataset's
# floor_offset_H instead.
GRID_INSET_H = 1e-6


@dataclass(frozen=True)
class CaseGeometry:
    """Body and ground on the mesh's own axes. `ground` is a coordinate along the up axis."""
    body_bounds: tuple[float, float, float, float, float, float]
    ground: float
    H: float
    up: Axis

    def height_above_ground(self, spec: ImageSpec) -> np.ndarray:
        """Height of each voxel layer along the up axis, shaped to broadcast against (nx, ny, nz)."""
        h = self.up.sign * (spec.axis(self.up.index) - self.ground)
        shape = [1, 1, 1]
        shape[self.up.index] = -1
        return h.reshape(shape)

    def image_spec(self, contract: Contract, domain: tuple[float, float, float, float, float, float],
                   floor_offset_H: float = GRID_INSET_H) -> ImageSpec:
        """
        Voxels of size H / n_per_H filling the domain (xmin, xmax, ymin, ymax, zmin, zmax), inset by
        GRID_INSET_H * H from its faces. Along the up axis the grid starts floor_offset_H * H above the ground,
        or at the domain's own face on the ground side when that face is above the ground.
        """
        dx = self.H / contract.n_per_H
        eps = GRID_INSET_H * self.H
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
                raise ValueError(f"empty domain along {'xyz'[a]}: {domain}, ground {self.ground}")
            n[a] = int(np.floor((end - start) / dx + 1e-9)) + 1
            # anchor at the ground side along the up axis, else at the low face
            lo[a] = end - (n[a] - 1) * dx if (a == self.up.index and self.up.sign < 0) else start
        return ImageSpec(tuple(float(v) for v in lo), dx, tuple(n))  # type: ignore[arg-type]


def case_geometry(body: vtkPolyData, ground: float, reference_length: float | None, up: Axis) -> CaseGeometry:
    b = body.GetBounds()
    j = up.index
    lowest, highest = (b[2 * j], b[2 * j + 1]) if up.sign > 0 else (-b[2 * j + 1], -b[2 * j])
    H = reference_length if reference_length is not None else highest - lowest
    if not H > 0:
        raise ValueError(f"reference length H must be > 0, got {H}")
    if lowest < up.sign * ground - 1e-6 * H:
        raise ValueError(f"body reaches below the ground along {up}: body {b}, ground {ground}")
    return CaseGeometry(tuple(b), float(ground), float(H), up)  # type: ignore[arg-type]


def body_inside(body: vtkPolyData, spec: ImageSpec) -> np.ndarray:
    """Voxels inside the closed body surface (scanline parity, robust to the surface's normals)."""
    n = spec.dims
    stencil = vtkPolyDataToImageStencil(input_data=body, output_origin=spec.origin, output_spacing=(spec.dx,) * 3,
                                        output_whole_extent=(0, n[0] - 1, 0, n[1] - 1, 0, n[2] - 1))
    to_image = vtkImageStencilToImage(input_connection=stencil.output_port, inside_value=1, outside_value=0,
                                      output_scalar_type=3)  # unsigned char
    to_image.Update()
    return vtk_to_numpy(to_image.GetOutput().GetPointData().GetScalars()).reshape(n, order="F").astype(bool)


def _exact_distance(body: vtkPolyData, spec: ImageSpec) -> np.ndarray:
    """Unsigned distance from every voxel of spec to the body's triangles (vtkImplicitPolyDataDistance)."""
    dist = vtkImplicitPolyDataDistance()
    dist.SetInput(body)
    sampler = vtkSampleFunction(implicit_function=dist, model_bounds=spec.bounds, sample_dimensions=spec.dims,
                                compute_normals=False, output_scalar_type=10)  # float
    sampler.Update()
    return np.abs(vtk_to_numpy(sampler.GetOutput().GetPointData().GetScalars())).reshape(spec.dims, order="F")


def signed_distance(body: vtkPolyData, spec: ImageSpec, band: float | None = None,
                    info: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Signed distance from each voxel to the body surface (mesh units, negative inside) and the inside mask.
    The sign comes from body_inside (scanline parity), because the implicit function's normal-based sign
    is wrong at some far-field points near sharp edges.

    band=None: the exact distance to the triangles everywhere (vtkImplicitPolyDataDistance), which is slow
    on large grids. band=b: exact only inside the body's bounding box grown by b; outside it, the Euclidean
    distance transform of the inside mask (distance to the nearest inside voxel centre), shifted by one
    offset so it meets the exact values on the band's boundary. Its error is a fraction of a voxel, small
    against distances > b. Training and inference both call this, so they get identical values.
    `info`, when given, receives the method and the offset. Raises ValueError if no voxel is inside the body.
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

    from scipy.ndimage import binary_erosion, distance_transform_edt
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
