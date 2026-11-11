"""Small synthetic meshes and fields for the tests."""

from __future__ import annotations

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkUnstructuredGrid
from vtkmodules.vtkFiltersCore import vtkAppendFilter
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter


def box_mesh(n=(9, 9, 9), lo=(-1.0, -1.0, 0.0), hi=(1.0, 1.0, 2.0), tets=True) -> vtkUnstructuredGrid:
    """A box of hexahedra, optionally split into tetrahedra."""
    spacing = tuple((hi[i] - lo[i]) / (n[i] - 1) for i in range(3))
    img = vtkImageData(dimensions=n, origin=lo, spacing=spacing)
    if tets:
        f = vtkDataSetTriangleFilter(input_data=img)
    else:
        f = vtkAppendFilter(input_data=img)
    f.Update()
    ug = vtkUnstructuredGrid()
    ug.ShallowCopy(f.GetOutput())
    return ug


def points(ug) -> np.ndarray:
    return vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)


def cell_centers(ug) -> np.ndarray:
    from vtkmodules.vtkFiltersCore import vtkCellCenters
    c = vtkCellCenters(input_data=ug)
    c.Update()
    return vtk_to_numpy(c.GetOutput().GetPoints().GetData()).astype(np.float64)


def add(ug, name, a, assoc="point"):
    v = numpy_to_vtk(np.ascontiguousarray(a.astype(np.float32)), deep=True)
    v.SetName(name)
    (ug.GetPointData() if assoc == "point" else ug.GetCellData()).AddArray(v)


def set_linear_velocity(ug, G, U0=(0.0, 0.0, 0.0), p=None, both=True, name="UMean", pname="pMean"):
    """u(x) = U0 + G x and pressure p(x) (or 0) at the points, and at the cell centers too when both."""
    G = np.asarray(G, dtype=float)
    P = points(ug)
    add(ug, name, np.asarray(U0) + P @ G.T)
    add(ug, pname, p(P) if p is not None else np.zeros(len(P)))
    if both:
        C = cell_centers(ug)
        add(ug, name, np.asarray(U0) + C @ G.T, "cell")
        add(ug, pname, p(C) if p is not None else np.zeros(len(C)), "cell")


def interior(ug, layers, lo=(-1.0, -1.0, 0.0), hi=(1.0, 1.0, 2.0), h=0.25):
    """Points at least `layers` cells (of size h) from every face of box_mesh's default box."""
    P = vtk_to_numpy(ug.GetPoints().GetData())
    d = layers * h - 1e-9
    return np.all((P >= np.array(lo) + d) & (P <= np.array(hi) - d), axis=1)
