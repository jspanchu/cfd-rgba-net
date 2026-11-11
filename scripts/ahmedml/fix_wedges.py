"""
Rewrite the AhmedML volume VTUs with their wedges in VTK's point order, in appended mode.

The OpenFOAM-written volume_<N>.vtu files have every VTK_WEDGE inside out: triangle (0,1,2)
faces away from triangle (3,4,5), the mirror of VTK's convention. This script downloads each
run's VTU into --staging, reorders each inverted wedge's points (0,1,2,3,4,5) -> (0,2,1,3,5,4),
verifies the result, writes <data>/run_<N>/volume_<N>.vtu in VTK XML appended mode (LZ4 by
default, see --setting), re-reads it, records <data>/run_<N>/volume_<N>.fixed.json and deletes
the staged original.

Points, point/cell/field data and every other cell are left untouched. Flat wedges are not
flipped, and other linear cells (hexahedra, tetrahedra, pyramids) are only reported, never
flipped. The OpenFOAM header comment (case, region, time, index), which VTK's reader drops, is
kept as one-value string arrays OpenFOAM_<key> in the field data.

A run is skipped when its volume_<N>.fixed.json has status "ok" and the output file exists with
the recorded size. A failed run is logged to <staging>/failures.log and its staged original is
kept. The exit status is 0 even when runs failed; check the log or the final "finished" line.

    python scripts/ahmedml/fix_wedges.py --runs 1-499 --data /data/cfd-rgba-net-data/ahmedML/data \
        --staging /data/cfd-rgba-net-data/ahmedML/staging --workers 2
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import shutil
import threading
import time
import traceback

from huggingface_hub import HfApi, hf_hub_download
import numpy as np

from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkCommand, vtkPoints, vtkStringArray
from vtkmodules.vtkCommonDataModel import (vtkHexahedron, vtkPyramid, vtkTetra, vtkUnstructuredGrid,
    vtkWedge, VTK_HEXAHEDRON, VTK_POLYHEDRON, VTK_PYRAMID, VTK_TETRA, VTK_WEDGE)
from vtkmodules.vtkFiltersCore import vtkExtractCells
from vtkmodules.vtkFiltersGeneral import vtkCellValidator
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter

from cfd_rgba_net.batch import parse_runs

REPO = "neashton/ahmedml"
# The wedge permutation that mirrors a wedge: reverses all five faces.
WEDGE_FLIP = [0, 2, 1, 3, 5, 4]
# |s| <= FLAT_TOL * h^3 (h = largest coordinate extent of the cell) counts as flat.
FLAT_TOL = 1e-6
# vtkCellValidator state bit for FacesAreOrientedIncorrectly.
FACES_ORIENTED_INCORRECTLY = 32

# Writer settings that --setting selects between. They were benchmarked on run 1.
SETTINGS = {
    "A":   dict(compressor="none", id_type=64),
    "A32": dict(compressor="none", id_type=32),
    "B":   dict(compressor="lz4", id_type=64),
    "C":   dict(compressor="zlib", level=1, id_type=64),
}
# B was the fastest cold read on run 1 (16.7 s vs 19.1 s for A, 66 s for the original) at 3.86 GB.
DEFAULT_SETTING = "B"


# --------------------------------------------------------------------------- #
# signed-volume tests of the linear cells

def _wedge_s(p):
    """Return a signed volume proxy per wedge: the base triangle's normal dotted with the base-to-top offset.

    Parameters
    ----------
    p : numpy.ndarray
        Wedge points, shape ``(n, 6, 3)``, float64.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``. Positive for a wedge in VTK's point order.
    """
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return np.einsum("ij,ij->i", n, p[:, 3:].mean(1) - p[:, :3].mean(1))

def _hex_s(p):
    """Return a signed volume proxy per hexahedron: the base quad's normal dotted with the base-to-top offset.

    The base quad is points (0,1,2,3) and the top quad is (4,5,6,7). The sign for VTK's order is in `REF`.

    Parameters
    ----------
    p : numpy.ndarray
        Hexahedron points, shape ``(n, 8, 3)``, float64.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``.
    """
    n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    return np.einsum("ij,ij->i", n, p[:, 4:].mean(1) - p[:, :4].mean(1))

def _tet_s(p):
    """Return the signed volume (times 6) per tetrahedron.

    The sign for VTK's order is in `REF`.

    Parameters
    ----------
    p : numpy.ndarray
        Tetrahedron points, shape ``(n, 4, 3)``, float64.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``.
    """
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])

def _pyramid_s(p):
    """Return a signed volume proxy per pyramid: the base quad's normal dotted with the base-to-apex offset.

    The base quad is points (0,1,2,3) and the apex is point 4. The sign for VTK's order is in `REF`.

    Parameters
    ----------
    p : numpy.ndarray
        Pyramid points, shape ``(n, 5, 3)``, float64.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``.
    """
    n = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])
    return np.einsum("ij,ij->i", n, p[:, 4] - p[:, :4].mean(1))

LINEAR = {  # type -> (name, vtk cell class, signed test)
    VTK_TETRA: ("tetra", vtkTetra, _tet_s),
    VTK_HEXAHEDRON: ("hexahedron", vtkHexahedron, _hex_s),
    VTK_WEDGE: ("wedge", vtkWedge, _wedge_s),
    VTK_PYRAMID: ("pyramid", vtkPyramid, _pyramid_s),
}

def _reference(cls):
    """Build VTK's reference cell of a type and read off its point count and faces.

    Parameters
    ----------
    cls : type
        The VTK cell class, for example ``vtkWedge``.

    Returns
    -------
    n : int
        Number of points of the cell.
    faces : list of list of int
        Each face as a list of local point ids.
    pc : numpy.ndarray
        The cell's parametric coordinates, shape ``(n, 3)``.
    """
    c = cls()
    n = c.GetNumberOfPoints()
    pc = np.array(c.GetParametricCoords()[:3 * n]).reshape(n, 3)
    for i in range(n):
        c.GetPointIds().SetId(i, i)
        c.GetPoints().SetPoint(i, *pc[i])
    faces = []
    for i in range(c.GetNumberOfFaces()):
        f = c.GetFace(i)
        faces.append([f.GetPointIds().GetId(j) for j in range(f.GetNumberOfPoints())])
    return n, faces, pc

# Per cell type: point count, faces and the sign of the signed test on VTK's own reference cell. They
# come from VTK at import time so the tests below cannot drift from VTK's point order and face layout.
REF = {}
for _t, (_, _cls, _test) in LINEAR.items():
    _n, _faces, _pc = _reference(_cls)
    _sign = float(np.sign(_test(_pc[None])[0]))
    assert _sign != 0
    REF[_t] = dict(n=_n, faces=_faces, sign=_sign)
assert REF[VTK_WEDGE]["sign"] > 0  # s > 0 is a correctly ordered wedge


def cells_of(types, offsets, ctype):
    """Find the cells of one fixed-size type and their positions in the connectivity array.

    Parameters
    ----------
    types : numpy.ndarray
        Cell type of every cell, shape ``(ncells,)``.
    offsets : numpy.ndarray
        Connectivity offsets, shape ``(ncells + 1,)``.
    ctype : int
        VTK cell type, a key of `LINEAR`.

    Returns
    -------
    ids : numpy.ndarray
        Ids of the cells of that type, in increasing order.
    idx : numpy.ndarray
        Shape ``(len(ids), npts)``. Indices into the connectivity array of each cell's points.

    Raises
    ------
    RuntimeError
        If a cell of that type does not have the type's point count.
    """
    ids = np.flatnonzero(types == ctype)
    n = REF[ctype]["n"]
    off = offsets[ids]
    if ids.size and not np.all(offsets[ids + 1] - off == n):
        raise RuntimeError(f"cells of type {ctype} without {n} points")
    return ids, off[:, None] + np.arange(n)


def signed_test(points, conn, idx, ctype, chunk=2_000_000):
    """Evaluate a type's signed test on cells, in chunks to bound memory.

    Parameters
    ----------
    points : numpy.ndarray
        Point coordinates, shape ``(npoints, 3)``.
    conn : numpy.ndarray
        Connectivity array.
    idx : numpy.ndarray
        Shape ``(m, npts)``. Indices into `conn` of each cell's points, as `cells_of` returns.
    ctype : int
        VTK cell type, a key of `LINEAR`.
    chunk : int, optional
        Number of cells evaluated at a time.

    Returns
    -------
    s : numpy.ndarray
        Shape ``(m,)``. The signed test times the type's reference sign, so ``s > 0`` means VTK's order.
    flat : numpy.ndarray
        Shape ``(m,)``, bool. True where ``|s| <= FLAT_TOL * h**3``, so the sign carries no information.
    """
    test = LINEAR[ctype][2]
    s = np.empty(len(idx))
    flat = np.empty(len(idx), bool)
    for a in range(0, len(idx), chunk):
        p = points[conn[idx[a:a + chunk]]].astype(np.float64)
        h = np.ptp(p, axis=1).max(1)
        s[a:a + chunk] = test(p) * REF[ctype]["sign"]
        flat[a:a + chunk] = np.abs(s[a:a + chunk]) <= FLAT_TOL * h ** 3
    return s, flat


# --------------------------------------------------------------------------- #
# shared-face check

def _face_records(arrs, owners, wedge_owned):
    """Compute a point-set key and a winding direction for every face.

    Parameters
    ----------
    arrs : list of numpy.ndarray
        One non-empty ``(m, k)`` int array of point ids per face size ``k``.
    owners : list of numpy.ndarray
        For each array in `arrs`, the ``(m,)`` ids of the cells that own its faces.
    wedge_owned : list of numpy.ndarray
        For each array in `arrs`, an ``(m,)`` bool array that is True where the owner is a wedge.

    Returns
    -------
    tuple of numpy.ndarray or None
        ``(keys, orient, owner, wedge_owned)`` over all faces, or None when `arrs` is empty. Two faces
        have the same point set exactly when their rows of ``keys`` match (the face size, then the
        sorted point ids, padded with -1). ``orient`` is the winding direction.
    """
    keys, orient, own, wed = [], [], [], []
    kmax = max(a.shape[1] for a in arrs) if arrs else 0
    for a, o, w in zip(arrs, owners, wedge_owned):
        k = a.shape[1]
        r = np.argmin(a, axis=1)
        rot = np.take_along_axis(a, (r[:, None] + np.arange(k)) % k, axis=1)
        key = np.full((len(a), kmax + 1), -1, np.int64)
        key[:, 0] = k
        key[:, 1:k + 1] = np.sort(a, axis=1)
        keys.append(key)
        # With the smallest id rotated to the front, comparing its two neighbours tells the winding
        # direction; reversing the winding swaps them. Two faces with one point set are wound the same
        # way exactly when their flags are equal.
        orient.append(rot[:, 1] < rot[:, -1])
        own.append(o)
        wed.append(w)
    if not keys:
        return None
    return np.concatenate(keys), np.concatenate(orient), np.concatenate(own), np.concatenate(wed)


def shared_face_check(grid, types, offsets, conn, chunk=4_000_000):
    """Check that cells sharing a face wind it oppositely.

    Matches every face whose points all belong to some wedge against the other faces with the same
    point set. A face shared by two cells must be wound oppositely by them; this counts the pairs
    wound the same way. Faces come from the linear cells (`LINEAR`) and from polyhedra.

    Parameters
    ----------
    grid : vtkUnstructuredGrid
        The grid. Only polyhedron faces are read from it directly.
    types : numpy.ndarray
        Cell type of every cell, shape ``(ncells,)``.
    offsets : numpy.ndarray
        Connectivity offsets, shape ``(ncells + 1,)``.
    conn : numpy.ndarray
        Connectivity array.
    chunk : int, optional
        Number of cells whose faces are gathered at a time.

    Returns
    -------
    dict
        Integer counts under these keys:

        - ``candidate_faces``: faces considered.
        - ``pairs``: point sets shared by exactly two faces.
        - ``same_wound_pairs``: pairs wound the same way, which is wrong.
        - ``same_wound_pairs_with_wedge``: those pairs in which at least one cell is a wedge.
        - ``unmatched_wedge_faces``: wedge faces that no other cell shares.
        - ``faces_in_more_than_2_cells``: point sets shared by more than two faces.
        - ``polyhedron_faces``: (cell, face) incidences of all polyhedra, candidate or not.

    Raises
    ------
    RuntimeError
        If a wedge does not have six points or a polyhedron face has fewer than three.
    """
    npts = grid.GetNumberOfPoints()
    _, wedge_idx = cells_of(types, offsets, VTK_WEDGE)
    mask = np.zeros(npts, bool)
    mask[conn[wedge_idx]] = True
    arrs, owners, wedge_owned = [], [], []
    for t in LINEAR:
        ids = np.flatnonzero(types == t)
        n, faces = REF[t]["n"], REF[t]["faces"]
        for a in range(0, len(ids), chunk):
            cid = ids[a:a + chunk]
            cpts = conn[offsets[cid][:, None] + np.arange(n)]
            m = mask[cpts]
            for f in faces:
                sel = m[:, f].all(1)
                if sel.any():
                    arrs.append(cpts[sel][:, f])
                    owners.append(cid[sel])
                    wedge_owned.append(np.full(sel.sum(), t == VTK_WEDGE))
    n_poly_faces = 0
    if np.any(types == VTK_POLYHEDRON):
        faces = grid.GetPolyhedronFaces()
        locs = grid.GetPolyhedronFaceLocations()
        f_off = vtk_to_numpy(faces.GetOffsetsArray()).astype(np.int64)
        f_conn = vtk_to_numpy(faces.GetConnectivityArray()).astype(np.int64)
        l_off = vtk_to_numpy(locs.GetOffsetsArray()).astype(np.int64)
        l_conn = vtk_to_numpy(locs.GetConnectivityArray()).astype(np.int64)
        owner = np.repeat(np.arange(len(l_off) - 1), np.diff(l_off))  # per (cell, face) incidence
        sizes = np.diff(f_off)
        # reduceat misreads empty or one-point slices, so reject such faces before using it.
        if np.any(sizes < 3):
            raise RuntimeError("polyhedron face with fewer than 3 points")
        all_in = np.logical_and.reduceat(mask[f_conn], f_off[:-1])
        sel = all_in[l_conn]
        fid, own = l_conn[sel], owner[sel]
        n_poly_faces = len(l_conn)
        for k in np.unique(sizes[fid]):
            s = sizes[fid] == k
            arrs.append(f_conn[f_off[fid[s]][:, None] + np.arange(k)])
            owners.append(own[s])
            wedge_owned.append(np.zeros(s.sum(), bool))
    rec = _face_records(arrs, owners, wedge_owned)
    out = dict(candidate_faces=0, pairs=0, same_wound_pairs=0, same_wound_pairs_with_wedge=0,
               unmatched_wedge_faces=0, faces_in_more_than_2_cells=0, polyhedron_faces=n_poly_faces)
    if rec is None:
        return out
    keys, orient, own, wed = rec
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    # Sorting by key puts the two faces of every count-2 group next to each other; take them in twos.
    order = np.argsort(inv, kind="stable")
    c = counts[inv[order]]
    first = order[(c == 2)][0::2]
    second = order[(c == 2)][1::2]
    same = orient[first] == orient[second]
    out.update(
        candidate_faces=int(len(keys)),
        pairs=int(len(first)),
        same_wound_pairs=int(same.sum()),
        same_wound_pairs_with_wedge=int((same & (wed[first] | wed[second])).sum()),
        unmatched_wedge_faces=int((wed & (counts[inv] == 1)).sum()),
        faces_in_more_than_2_cells=int((counts > 2).sum()),
    )
    return out


# --------------------------------------------------------------------------- #
# vtkCellValidator on the wedges

def _np(arr):
    """Convert a VTK array to numpy.

    Parameters
    ----------
    arr : vtkDataArray
        The array to convert.

    Returns
    -------
    numpy.ndarray
        A view of the array's memory when `vtk_to_numpy` accepts it. Otherwise a slow copy, which
        works for one-component arrays only.
    """
    try:
        return vtk_to_numpy(arr)
    except Exception:
        return np.array([arr.GetTuple1(i) for i in range(arr.GetNumberOfTuples())])


def validate_wedges(grid, wedge_ids):
    """Run vtkCellValidator on the wedges and summarise the validity states.

    Parameters
    ----------
    grid : vtkUnstructuredGrid
        The grid holding the wedges.
    wedge_ids : numpy.ndarray
        Ids of the wedge cells, sorted and unique, as `cells_of` returns them. `vtkExtractCells` is told
        to rely on that.

    Returns
    -------
    dict
        ``{}`` when there are no wedges. Otherwise ``histogram`` (state as a string -> count),
        ``faces_oriented_incorrectly`` (count of wedges with that state bit set) and ``bad`` (a list of
        ``(cell id, state)`` for those wedges).

    Raises
    ------
    RuntimeError
        If the extraction does not return every wedge.
    """
    if not len(wedge_ids):
        return {}
    ex = vtkExtractCells()
    ex.SetInputData(grid)
    ids = np.ascontiguousarray(wedge_ids, np.int64)
    ex.SetCellIds(ids, len(ids))
    ex.SetAssumeSortedAndUniqueIds(True)
    v = vtkCellValidator()
    v.SetInputConnection(ex.GetOutputPort())
    v.Update()
    out = v.GetOutput()
    if out.GetNumberOfCells() != len(wedge_ids):
        raise RuntimeError("vtkExtractCells did not return every wedge")
    states = _np(out.GetCellData().GetArray("ValidityState")).astype(np.int64)
    vals, cnt = np.unique(states, return_counts=True)
    bad = (states & FACES_ORIENTED_INCORRECTLY) != 0
    return dict(histogram={str(int(a)): int(b) for a, b in zip(vals, cnt)},
                faces_oriented_incorrectly=int(bad.sum()),
                bad=list(zip(np.asarray(wedge_ids)[bad].tolist(), states[bad].tolist())))


def validate_one(points6):
    """Run vtkCellValidator on one wedge.

    Parameters
    ----------
    points6 : numpy.ndarray
        The wedge's points in cell order, shape ``(6, 3)``.

    Returns
    -------
    int
        The validity state bit mask.
    """
    ug = vtkUnstructuredGrid()
    P = vtkPoints()
    for q in points6:
        P.InsertNextPoint(*q)
    ug.SetPoints(P)
    ug.InsertNextCell(VTK_WEDGE, 6, list(range(6)))
    v = vtkCellValidator()
    v.SetInputData(ug)
    v.Update()
    return int(v.GetOutput().GetCellData().GetArray("ValidityState").GetTuple1(0))


def classify_misoriented(grid, bad):
    """Split the wedges vtkCellValidator flags as FacesAreOrientedIncorrectly in two groups.

    A wedge is a real orientation error when the validator accepts its mirror. It is ambiguous when the
    validator rejects it in both orders, which happens for distorted cells (for example a
    near-degenerate triangle with twisted sides) whose orientation it cannot judge.

    Parameters
    ----------
    grid : vtkUnstructuredGrid
        The grid holding the wedges.
    bad : list of tuple of int
        ``(cell id, state)`` for each flagged wedge, as `validate_wedges` returns under ``bad``.

    Returns
    -------
    real : list of dict
        The real errors.
    ambiguous : list of dict
        The ambiguous ones. Both lists hold dicts with ``id``, ``state``, ``mirrored_state`` (the
        validator state of the mirror) and ``s`` (the wedge's signed test).
    """
    offsets = vtk_to_numpy(grid.GetCells().GetOffsetsArray())
    conn = vtk_to_numpy(grid.GetCells().GetConnectivityArray())
    points = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float64)
    real, ambiguous = [], []
    for cid, state in bad:
        p = points[conn[offsets[cid]:offsets[cid] + 6]]
        mirrored = validate_one(p[WEDGE_FLIP])
        d = dict(id=int(cid), state=int(state), mirrored_state=mirrored, s=float(_wedge_s(p[None])[0]))
        (ambiguous if mirrored & FACES_ORIENTED_INCORRECTLY else real).append(d)
    return real, ambiguous


# --------------------------------------------------------------------------- #
# checksums

def _sha(a):
    """Hash an array's raw bytes.

    Parameters
    ----------
    a : numpy.ndarray
        The array. Its dtype is not part of the hash, so equal bytes of different dtypes collide.

    Returns
    -------
    str
        The SHA-256 hex digest.
    """
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def _array_summary(arr):
    """Describe one array so two arrays can be compared without keeping their data.

    Parameters
    ----------
    arr : vtkAbstractArray
        A numeric or string array.

    Returns
    -------
    dict
        ``dtype``, ``components``, ``tuples`` and ``sha256``. A string array also has ``values`` (None
        when it holds more than 16). A non-empty float or integer array also has ``sum``, ``min`` and
        ``max``.
    """
    if arr.IsA("vtkStringArray"):
        values = [arr.GetValue(i) for i in range(arr.GetNumberOfValues())]
        return dict(dtype="string", components=arr.GetNumberOfComponents(), tuples=arr.GetNumberOfTuples(),
                    values=values if len(values) <= 16 else None,
                    sha256=hashlib.sha256("\0".join(values).encode()).hexdigest())
    a = _np(arr)
    d = dict(dtype=str(a.dtype), components=arr.GetNumberOfComponents(), tuples=arr.GetNumberOfTuples(),
             sha256=_sha(a))
    if a.size and a.dtype.kind in "fiu":
        d.update(sum=float(np.sum(a, dtype=np.float64)), min=float(np.nanmin(a)), max=float(np.nanmax(a)))
    return d


def summarize(grid):
    """Summarise everything that identifies a grid's content, for comparing two grids.

    Parameters
    ----------
    grid : vtkUnstructuredGrid
        The grid.

    Returns
    -------
    dict
        JSON-serialisable. It holds the point and cell counts, a summary of the point coordinates and of
        every point, cell and field data array, hashes of the cell types, offsets and connectivity, and
        hashes of the polyhedron face arrays when present.
    """
    d = dict(points=grid.GetNumberOfPoints(), cells=grid.GetNumberOfCells(),
             point_coords=_array_summary(grid.GetPoints().GetData()))
    for assoc, data in (("point_data", grid.GetPointData()), ("cell_data", grid.GetCellData()),
                        ("field_data", grid.GetFieldData())):
        d[assoc] = {data.GetArrayName(i): _array_summary(data.GetAbstractArray(i))
                    for i in range(data.GetNumberOfArrays())}
    cells = grid.GetCells()
    d["types"] = _sha(_np(grid.GetCellTypesArray()).astype(np.uint8))
    d["offsets"] = _sha(vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int64))
    d["connectivity"] = _sha(vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64))
    faces, locs = grid.GetPolyhedronFaces(), grid.GetPolyhedronFaceLocations()
    for name, ca in (("polyhedron_faces", faces), ("polyhedron_face_locations", locs)):
        if ca is not None:
            d[name] = dict(offsets=_sha(vtk_to_numpy(ca.GetOffsetsArray()).astype(np.int64)),
                           connectivity=_sha(vtk_to_numpy(ca.GetConnectivityArray()).astype(np.int64)))
    return d


def diff_summaries(a, b, ignore=()):
    """List where two summaries differ.

    NaN never equals NaN, so an array that contains NaN always reports a difference through its ``sum``,
    ``min`` and ``max``.

    Parameters
    ----------
    a : dict
        A summary from `summarize`.
    b : dict
        The summary to compare it with.
    ignore : collection of str, optional
        Dotted paths, such as ``"connectivity"`` or ``"point_data.p"``, to skip along with everything
        beneath them.

    Returns
    -------
    list of str
        The dotted paths whose values differ or that exist on one side only.
    """
    bad = []
    def walk(x, y, path):
        if path in ignore:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{path}.{k}" if path else k)
        elif x != y:
            bad.append(path)
    walk(a, b, "")
    return bad


# --------------------------------------------------------------------------- #
# the OpenFOAM header comment, e.g. <!-- case='run1' region='region0' time='79.9998' index='133333' -->

OPENFOAM_PREFIX = "OpenFOAM_"


def parse_openfoam_header(head: bytes):
    """Read the ``key='value'`` pairs of the XML comment that OpenFOAM writes before ``<VTKFile``.

    Parameters
    ----------
    head : bytes
        The start of an OpenFOAM-written VTU.

    Returns
    -------
    dict of str to str
        The pairs of the first XML comment, or ``{}`` when there is none before ``<VTKFile``.
    """
    text = head.decode("latin-1")
    a = text.find("<!--")
    b = text.find("-->", a)
    if a < 0 or b < 0 or a > text.find("<VTKFile"):
        return {}
    return dict(re.findall(r"(\w+)='([^']*)'", text[a + 4:b]))


def read_openfoam_header(path, n=4096):
    """Read the OpenFOAM header comment from a VTU file.

    Parameters
    ----------
    path : str or pathlib.Path
        The VTU file.
    n : int, optional
        Number of bytes to read from the start of the file. The header must fit in them.

    Returns
    -------
    dict of str to str
        As `parse_openfoam_header` returns.
    """
    with open(path, "rb") as f:
        return parse_openfoam_header(f.read(n))


def add_openfoam_field_data(grid, meta):
    """Store each header key as a one-value string array OpenFOAM_<key> in the grid's field data.

    Parameters
    ----------
    grid : vtkUnstructuredGrid
        The grid to add the arrays to, in place.
    meta : dict of str to str
        The header pairs, as `parse_openfoam_header` returns.
    """
    fd = grid.GetFieldData()
    for k, v in meta.items():
        s = vtkStringArray()
        s.SetName(OPENFOAM_PREFIX + k)
        s.InsertNextValue(v)
        fd.AddArray(s)


# --------------------------------------------------------------------------- #
# reading and writing

def read_vtu(path):
    """Read a VTU file and fail on any reader error.

    Parameters
    ----------
    path : str or pathlib.Path
        The VTU file.

    Returns
    -------
    vtkUnstructuredGrid
        A shallow copy of the reader's output.

    Raises
    ------
    RuntimeError
        If the reader reports an error.
    """
    r = vtkXMLUnstructuredGridReader()
    errs = []
    r.AddObserver(vtkCommand.ErrorEvent, lambda o, e: errs.append(e))
    r.SetFileName(str(path))
    r.Update()
    if errs or r.GetErrorCode():
        raise RuntimeError(f"reading {path} failed: error code {r.GetErrorCode()} {errs}")
    out = vtkUnstructuredGrid()
    out.ShallowCopy(r.GetOutput())
    return out


def write_vtu(grid, path, setting):
    """Write a VTU in appended mode with one of the `SETTINGS`.

    Parameters
    ----------
    grid : vtkUnstructuredGrid
        The grid to write.
    path : str or pathlib.Path
        The output file.
    setting : str
        A key of `SETTINGS`.

    Raises
    ------
    KeyError
        If `setting` is not a key of `SETTINGS`.
    RuntimeError
        If the writer reports an error.
    """
    s = SETTINGS[setting]
    w = vtkXMLUnstructuredGridWriter()
    errs = []
    w.AddObserver(vtkCommand.ErrorEvent, lambda o, e: errs.append(e))
    w.SetInputData(grid)
    w.SetFileName(str(path))
    w.SetDataModeToAppended()
    w.SetEncodeAppendedData(False)  # raw bytes in the appended section, not base64
    w.SetHeaderTypeToUInt64()
    w.SetIdTypeToInt64() if s["id_type"] == 64 else w.SetIdTypeToInt32()
    if s["compressor"] == "none":
        w.SetCompressorTypeToNone()
    elif s["compressor"] == "lz4":
        w.SetCompressorTypeToLZ4()
    elif s["compressor"] == "zlib":
        w.SetCompressorTypeToZLib()
        w.SetCompressionLevel(s["level"])
    if not w.Write() or errs:
        raise RuntimeError(f"writing {path} failed: {errs}")


def sha256_file(path, block=1 << 24):
    """Hash a file in blocks.

    This deliberately does not reuse `cfd_rgba_net.io.sha256`. That function caches the digest in a
    sidecar file, which would litter the staging folder, and a cached digest could hide a corrupt
    download from the check against Hugging Face's hash.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to hash.
    block : int, optional
        Number of bytes read at a time.

    Returns
    -------
    str
        The SHA-256 hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(block):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# the fix

def fix_wedges(grid):
    """Reorder the points of every inverted wedge with `WEDGE_FLIP`, in place.

    A wedge is inverted when its signed test is negative and it is not flat. Flat wedges are left as they
    are, and the other linear cells are tested and reported but never changed.

    Parameters
    ----------
    grid : vtkUnstructuredGrid
        The grid. Its connectivity array is rewritten in place through numpy.

    Returns
    -------
    report : dict
        ``cell_types`` (type -> count) and ``linear_cells`` (per type name: ``count``, ``inside_out``,
        ``flat`` and ``first_inside_out_ids``). When the grid has wedges it also has ``wedges_flipped``
        and ``wedges_inside_out_after``, the count that a re-test of the modified grid still finds
        inside out.
    wedge_ids : numpy.ndarray
        Ids of all wedge cells, or an empty array when there are none.

    Raises
    ------
    RuntimeError
        If numpy cannot write to the connectivity array or a linear cell has the wrong point count.
    """
    types = _np(grid.GetCellTypesArray()).astype(np.uint8)
    cells = grid.GetCells()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    conn = vtk_to_numpy(cells.GetConnectivityArray())
    if not conn.flags.writeable:
        raise RuntimeError("connectivity is not writable through numpy")
    points = vtk_to_numpy(grid.GetPoints().GetData())

    vals, cnt = np.unique(types, return_counts=True)
    report = dict(cell_types={str(int(a)): int(b) for a, b in zip(vals, cnt)}, linear_cells={})
    for t, (name, _, _) in LINEAR.items():
        ids, idx = cells_of(types, offsets, t)
        if not len(ids):
            continue
        s, flat = signed_test(points, conn, idx, t)
        inv = (s < 0) & ~flat
        report["linear_cells"][name] = dict(count=int(len(ids)), inside_out=int(inv.sum()), flat=int(flat.sum()),
                                            first_inside_out_ids=ids[inv][:10].tolist())
        if t == VTK_WEDGE:
            wedge_ids, wedge_idx, wedge_inv = ids, idx, inv  # only bound when the grid has wedges
    if "wedge" not in report["linear_cells"]:
        return report, np.zeros(0, np.int64)

    sel = wedge_idx[wedge_inv]
    conn[sel] = conn[sel][:, WEDGE_FLIP]
    cells.GetConnectivityArray().Modified()
    cells.Modified()
    grid.Modified()
    report["wedges_flipped"] = int(wedge_inv.sum())

    # Re-test through a fresh array fetched from the grid, so the check sees what VTK will write.
    conn2 = vtk_to_numpy(grid.GetCells().GetConnectivityArray())
    s, flat = signed_test(points, conn2, wedge_idx, VTK_WEDGE)
    report["wedges_inside_out_after"] = int(((s < 0) & ~flat).sum())
    return report, wedge_ids


def convert(run, src, data, setting, expected_sha256=None):
    """Fix one run's VTU and write the fixed file and its report.

    The steps are:

    1. Hash the source and compare it with Hugging Face's hash, when given.
    2. Read the source, add the OpenFOAM header to the field data and summarise it.
    3. Run the shared-face check.
    4. Flip the inverted wedges (`fix_wedges`).
    5. Verify before writing: vtkCellValidator on the wedges, the shared-face check again, the re-test of
       the flip, a summary diff that may differ in the connectivity only, and a check that the
       connectivity did change when wedges were flipped. Any failure raises with all the problems.
    6. Write ``volume_<run>.vtu.tmp``, re-read it and require its summary to equal the fixed grid's.
    7. Move the output into place, then write the report atomically.

    Parameters
    ----------
    run : int
        The run number.
    src : str or pathlib.Path
        The original VTU.
    data : str or pathlib.Path
        Dataset root. The output goes to ``<data>/run_<run>/``.
    setting : str
        A key of `SETTINGS`.
    expected_sha256 : str, optional
        The source's expected SHA-256. The check is skipped when None.

    Returns
    -------
    dict
        The report, as written to ``volume_<run>.fixed.json``.

    Raises
    ------
    FileNotFoundError
        If `src` does not exist.
    RuntimeError
        If the hash does not match, a read or write fails, or verification finds a problem. The output
        and report are untouched in that case, but a ``.tmp`` file can remain once step 6 has started.
    """
    T = {}  # seconds per step, stored in the report
    t0 = time.time()
    def lap(name):
        nonlocal t0
        T[name] = round(time.time() - t0, 2)
        t0 = time.time()
    def log(msg):
        print(f"[fix] run_{run}: {msg}", flush=True)

    src = Path(src)
    dst_dir = Path(data) / f"run_{run}"
    dst = dst_dir / f"volume_{run}.vtu"
    tmp = dst_dir / f"volume_{run}.vtu.tmp"
    rep_path = dst_dir / f"volume_{run}.fixed.json"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # The report reaches the disk only after a successful run, where its status is "ok".
    rep = dict(run=run, status="running", source=dict(file=str(src), bytes=src.stat().st_size))
    rep["source"]["sha256"] = sha256_file(src)
    if expected_sha256:
        rep["source"]["sha256_expected"] = expected_sha256
        if rep["source"]["sha256"] != expected_sha256:
            raise RuntimeError(f"sha256 of {src} does not match Hugging Face's ({expected_sha256})")
    lap("sha256")

    grid = read_vtu(src)
    lap("read_source")
    rep["openfoam"] = read_openfoam_header(src)
    add_openfoam_field_data(grid, rep["openfoam"])
    log(f"read {grid.GetNumberOfPoints()} points, {grid.GetNumberOfCells()} cells in {T['read_source']}s")
    # Summarise after adding the header arrays, so the only difference the fix may cause is the connectivity.
    before = summarize(grid)
    lap("summarize_source")

    # types and offsets stay valid after the fix: it permutes points within cells and nothing else.
    types = _np(grid.GetCellTypesArray()).astype(np.uint8)
    offsets = vtk_to_numpy(grid.GetCells().GetOffsetsArray())
    conn = vtk_to_numpy(grid.GetCells().GetConnectivityArray())
    rep["shared_faces_before"] = shared_face_check(grid, types, offsets, conn)
    lap("shared_faces_before")

    fix, wedge_ids = fix_wedges(grid)
    rep.update(fix)
    lap("fix")
    log(f"flipped {fix.get('wedges_flipped', 0)} of {len(wedge_ids)} wedges; linear cells {fix['linear_cells']}")

    # ---- verify before writing ---- #
    errors = []
    vw = validate_wedges(grid, wedge_ids)
    real, ambiguous = classify_misoriented(grid, vw.pop("bad", []))
    vw.update(misoriented=real, ambiguous=ambiguous)
    rep["validator_wedges"] = vw
    if real:
        errors.append(f"{len(real)} wedges FacesAreOrientedIncorrectly whose mirror is valid: {real[:5]}")
    if ambiguous:
        log(f"{len(ambiguous)} wedges FacesAreOrientedIncorrectly in both orders (distorted; kept): {ambiguous}")
    lap("validator")
    # Fetch the connectivity again: the flip rewrote it in place.
    sf = shared_face_check(grid, types, offsets, vtk_to_numpy(grid.GetCells().GetConnectivityArray()))
    rep["shared_faces_after"] = sf
    if sf["same_wound_pairs"]:
        errors.append(f"{sf['same_wound_pairs']} shared faces wound the same way by both cells")
    if fix.get("wedges_inside_out_after", 0):
        errors.append(f"{fix['wedges_inside_out_after']} wedges still inside out")
    lap("shared_faces_after")
    fixed = summarize(grid)
    d = diff_summaries(before, fixed, ignore=("connectivity",))
    if d:
        errors.append(f"fix changed {d}")
    if fix.get("wedges_flipped", 0) and fixed["connectivity"] == before["connectivity"]:
        errors.append("connectivity unchanged after flipping wedges")
    lap("summarize_fixed")
    if errors:
        raise RuntimeError("; ".join(errors))

    # ---- write, re-read, compare ---- #
    write_vtu(grid, tmp, setting)
    lap("write")
    del grid  # release the multi-GB grid before reading the output back
    reread = read_vtu(tmp)
    lap("reread")
    after = summarize(reread)
    d = diff_summaries(fixed, after)
    if d:
        raise RuntimeError(f"re-read {tmp} differs from the fixed grid in {d}")
    lap("compare")
    del reread
    # Publish the VTU before its report: is_done trusts a report only when its output file exists.
    os.replace(tmp, dst)

    rep.update(status="ok", output=dict(file=str(dst), bytes=dst.stat().st_size,
                                        writer=dict(setting=setting, data_mode="appended", encode=False,
                                                    header_type="UInt64", **SETTINGS[setting])),
               summary=fixed, timings_s=T, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rtmp = rep_path.with_suffix(".json.tmp")
    rtmp.write_text(json.dumps(rep, indent=2))
    os.replace(rtmp, rep_path)
    log(f"ok: {rep['source']['bytes']/1e9:.2f} GB -> {rep['output']['bytes']/1e9:.2f} GB; "
        f"shared same-wound faces {rep['shared_faces_before']['same_wound_pairs']} -> {sf['same_wound_pairs']}; {T}")
    return rep


def is_done(run, data):
    """Tell whether a run has already been converted.

    Parameters
    ----------
    run : int
        The run number.
    data : str or pathlib.Path
        Dataset root holding the ``run_<N>/`` folders.

    Returns
    -------
    bool
        True when ``volume_<run>.fixed.json`` has status ``ok`` and the output file it names exists with the
        recorded size. A missing or unreadable report counts as not done. The output's content is not checked.
    """
    p = Path(data) / f"run_{run}" / f"volume_{run}.fixed.json"
    try:
        rep = json.loads(p.read_text())
        out = Path(rep["output"]["file"])
        return rep["status"] == "ok" and out.exists() and out.stat().st_size == rep["output"]["bytes"]
    except Exception:
        return False


def expected_sha256(run):
    """Look up a run's SHA-256 on Hugging Face.

    Parameters
    ----------
    run : int
        The run number.

    Returns
    -------
    str or None
        The LFS SHA-256 of ``run_<run>/volume_<run>.vtu``. None when the lookup fails or the file has no
        LFS hash, in which case `convert` skips the hash check.
    """
    try:
        info = HfApi().get_paths_info(REPO, [f"run_{run}/volume_{run}.vtu"], repo_type="dataset", expand=True)
        return info[0].lfs.sha256 if info and info[0].lfs else None
    except Exception as e:
        print(f"[dl ] run_{run}: no sha256 from Hugging Face ({e})", flush=True)
        return None


def download(run, staging):
    """Fetch ``run_<run>/volume_<run>.vtu`` from `REPO` into staging, unless a non-empty copy is there.

    Parameters
    ----------
    run : int
        The run number.
    staging : str or pathlib.Path
        Folder that receives ``run_<run>/volume_<run>.vtu``.

    Returns
    -------
    pathlib.Path
        The staged file.
    """
    rel = f"run_{run}/volume_{run}.vtu"
    target = Path(staging) / rel
    if target.exists() and target.stat().st_size > 0:
        return target
    t0 = time.time()
    hf_hub_download(repo_id=REPO, filename=rel, local_dir=staging, repo_type="dataset")
    print(f"[dl ] {REPO}: {rel}  {target.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)
    return target


def _convert_job(run, src, data, setting, sha):
    """Run `convert` in a worker process and report the outcome instead of raising.

    The traceback is formatted here so the parent process can log the worker's stack with the failure.

    Parameters
    ----------
    run : int
        The run number.
    src : str
        The staged VTU.
    data : str
        Dataset root.
    setting : str
        A key of `SETTINGS`.
    sha : str or None
        The expected SHA-256 of `src`.

    Returns
    -------
    tuple
        ``(report, None)`` on success, ``(None, traceback text)`` on failure.
    """
    try:
        return convert(run, src, data, setting, sha), None
    except Exception:
        return None, traceback.format_exc()


def main():
    """Run the command line interface: download, convert and clean up the requested runs.

    A downloader thread stages one VTU at a time and queues it. A process pool converts the queued runs
    with `_convert_job`, and the staged original is deleted after a successful conversion. Runs that
    `is_done` already reports are skipped. Downloads stop after ``--max-failures`` failed runs or when the
    staging disk has less than ``--min-free-gb`` free; runs already queued still finish.

    Returns
    -------
    None
        The exit status is 0 even when runs failed; see ``<staging>/failures.log``.

    Raises
    ------
    ValueError
        If ``--runs`` has a backwards range or a non-numeric entry (from `parse_runs`).
    SystemExit
        If argparse rejects the arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="'1,5,7', '1-499', '1-3,10', or a CSV file with a 'run' column")
    ap.add_argument("--data", required=True, help="dataset root holding run_<N>/ folders")
    ap.add_argument("--staging", required=True, help="where the original VTUs are downloaded to")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--setting", default=DEFAULT_SETTING, choices=sorted(SETTINGS))
    ap.add_argument("--max-failures", type=int, default=5, help="stop downloading after this many failures")
    ap.add_argument("--min-free-gb", type=float, default=50, help="stop downloading below this much free space")
    args = ap.parse_args()
    runs = parse_runs(args.runs)
    todo = [r for r in runs if not is_done(r, args.data)]
    print(f"{len(runs)} runs requested, {len(todo)} to do", flush=True)
    staging = Path(args.staging)
    staging.mkdir(parents=True, exist_ok=True)
    failures_log = staging / "failures.log"

    # One permit per run that is staged or being converted: every worker busy plus one download ahead.
    # This bounds the disk the staged originals use.
    slots = threading.Semaphore(args.workers + 1)
    stop = threading.Event()
    q = queue.Queue()

    def downloader():
        for r in todo:
            slots.acquire()
            if stop.is_set():
                break
            free = shutil.disk_usage(staging).free / 1e9
            if free < args.min_free_gb:
                print(f"[dl ] only {free:.0f} GB free; stopping downloads", flush=True)
                break
            try:
                sha = expected_sha256(r)
                download(r, staging)
                q.put((r, sha))
            except Exception as e:
                print(f"[dl ] run_{r}: {e}", flush=True)
                q.put((-r, None))  # a negative run number marks a failed download; runs start at 1
        q.put(None)  # tells the main loop that no more runs will come
    threading.Thread(target=downloader, daemon=True).start()

    results = {}
    lock = threading.Lock()
    def record(r, ok, msg=""):
        with lock:
            results[r] = ok
            if not ok:
                with failures_log.open("a") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} run_{r}\n{msg}\n")
            bad = sorted(k for k, v in results.items() if not v)
            print(f"done: {sum(results.values())}/{len(results)} ok of {len(todo)}; failed: {bad}", flush=True)
            if len(bad) >= args.max_failures:
                stop.set()

    def finished(r, src, fut):  # runs in the pool's result-handling thread
        try:
            rep, err = fut.result()
        except Exception:
            rep, err = None, traceback.format_exc()
        if rep is not None:
            src.unlink(missing_ok=True)
            record(r, True)
        else:
            print(f"[fix] run_{r}: FAILED\n{err}", flush=True)
            record(r, False, err)
        slots.release()

    # Spawn, so the workers do not fork this multithreaded process. One task per worker process, so the
    # memory VTK holds for a multi-GB grid goes back to the system after every run.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, max_tasks_per_child=1) as pool:
        while (item := q.get()) is not None:
            r, sha = item
            if r < 0:
                record(-r, False, "download failed")
                slots.release()
                continue
            src = staging / f"run_{r}" / f"volume_{r}.vtu"
            fut = pool.submit(_convert_job, r, str(src), args.data, args.setting, sha)
            fut.add_done_callback(lambda f, r=r, src=src: finished(r, src, f))
    bad = sorted(k for k, v in results.items() if not v)
    print(f"finished: {sum(results.values())} ok, {len(bad)} failed {bad}", flush=True)


if __name__ == "__main__":
    main()
