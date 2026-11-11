"""Writers (Zarr v3 case store, vti) and provenance (code version, input hashes)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter

from . import CONTRACT_VERSION, LABEL_RULES_VERSION, __version__
from .geometry import ImageSpec


def git_state(repo: Path | None = None) -> dict:
    """Record the git commit of the code that built a case.

    The default repository is two levels above this package, which is the repository root in a source
    checkout. For an installed wheel that location is outside the project, so the result is ``None`` or,
    if the environment sits inside some other repository, that repository's commit.

    Parameters
    ----------
    repo : pathlib.Path or None, optional
        The repository. Defaults to the repository root of a source checkout.

    Returns
    -------
    dict
        ``commit`` (the HEAD hash) and ``dirty`` (true if tracked files have uncommitted changes;
        untracked files are ignored). Both are ``None`` if git is missing or ``repo`` is not a repository.
    """
    repo = repo or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return dict(commit=commit, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)


def sha256(path: str | Path, chunk: int = 1 << 24) -> str:
    """Compute the SHA-256 of a file, caching it next to the file.

    The volume files are tens of GB and do not change between builds, so the digest is cached in
    ``<file>.sha256`` together with the file's size and modification time, and reused while both match.
    A file that changes without changing size or mtime would return a stale digest. Writing the cache
    can fail (for example in a read-only directory); that is ignored and the digest is still returned.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to hash.
    chunk : int, optional
        Number of bytes read at a time.

    Returns
    -------
    str
        The hex digest.
    """
    path = Path(path)
    st = path.stat()
    cache = path.with_name(path.name + ".sha256")
    key = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
    try:
        c = json.loads(cache.read_text())
        if {k: c.get(k) for k in key} == key:
            return c["sha256"]
    except (OSError, ValueError, KeyError):
        pass
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    digest = h.hexdigest()
    try:
        tmp = cache.with_name(cache.name + f".{os.getpid()}")
        tmp.write_text(json.dumps(dict(**key, sha256=digest)))
        tmp.replace(cache)
    except OSError:
        pass
    return digest


def provenance(inputs: dict[str, Path]) -> dict:
    """Collect what identifies the code and the inputs that built a case.

    Parameters
    ----------
    inputs : dict[str, pathlib.Path]
        Input files by role. An entry that is ``None``, missing, or not a regular file is skipped without
        an error.

    Returns
    -------
    dict
        ``package_version``, ``contract_version``, ``label_rules_version``, ``git`` (see ``git_state``),
        and ``inputs``: the path and SHA-256 of each input that was recorded.
    """
    return dict(
        package_version=__version__,
        contract_version=CONTRACT_VERSION,
        label_rules_version=LABEL_RULES_VERSION,
        git=git_state(),
        inputs={k: dict(path=str(p), sha256=sha256(p)) for k, p in inputs.items() if p is not None and Path(p).is_file()},
    )


CHUNK = 64     # voxels per chunk edge: a training patch reads only the chunks it overlaps
SHARD = 256    # voxels per shard edge: chunks are packed into shards, a few files per array
COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")


def _put(group, name: str, a: np.ndarray):
    """Store an array in a Zarr group, chunked and sharded along the three spatial axes.

    Leading axes (channels) are kept whole in every chunk.

    Parameters
    ----------
    group : zarr.Group
        The group to create the array in.
    name : str
        The array's name.
    a : numpy.ndarray
        The values, shaped ``(..., nx, ny, nz)``.
    """
    lead = a.shape[:-3]
    arr = group.create_array(name, shape=a.shape, dtype=a.dtype, chunks=lead + (CHUNK,) * 3,
                             shards=lead + (SHARD,) * 3, compressors=COMPRESSOR)
    arr[...] = a


def jsonable(obj):
    """Convert an object to plain JSON types.

    NumPy scalars become floats, so a NumPy boolean becomes ``1.0`` or ``0.0``, not ``true`` or ``false``.
    NumPy arrays are not supported and raise ``TypeError``. ``NaN`` and infinities pass through as the
    non-standard ``NaN`` and ``Infinity`` tokens.

    Parameters
    ----------
    obj : object
        Dicts, lists, tuples, strings, numbers, ``None`` and NumPy scalars, nested to any depth.

    Returns
    -------
    object
        The same structure in plain Python types, with tuples as lists.
    """
    return json.loads(json.dumps(obj, default=float))


def save_case_zarr(path: Path, spec: ImageSpec, arrays: dict[str, np.ndarray],
                   groups: dict[str, dict[str, np.ndarray]], attrs: dict):
    """Write one case as a Zarr v3 store.

    The store goes to ``<path>.tmp`` and is renamed to ``path`` at the end, so a store that exists is
    complete. An existing store is renamed aside to ``<path>.old`` first and deleted only once the new
    one is in place. A crash between the two renames leaves ``.old`` and ``.tmp`` and no store at ``path``.

    Parameters
    ----------
    path : pathlib.Path
        The store's location. Its parent directory must exist.
    spec : ImageSpec
        The voxel grid. Saved in the root attributes as ``grid``.
    arrays : dict[str, numpy.ndarray]
        Arrays stored at the root, shaped ``(..., nx, ny, nz)``.
    groups : dict[str, dict[str, numpy.ndarray]]
        Sub-groups by name, each holding arrays shaped ``(..., nx, ny, nz)``.
    attrs : dict
        Root attributes, converted with ``jsonable``. Must not have a ``grid`` key.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    old = path.with_name(path.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    root = zarr.open_group(tmp, mode="w", zarr_format=3)
    for name, a in arrays.items():
        _put(root, name, a)
    for gname, members in groups.items():
        g = root.create_group(gname)
        for name, a in members.items():
            _put(g, name, a)
    root.attrs.update(jsonable(dict(grid=spec.to_dict(), **attrs)))
    if path.exists():
        shutil.rmtree(old, ignore_errors=True)
        path.rename(old)
    tmp.rename(path)
    shutil.rmtree(old, ignore_errors=True)


def save_vti(path: Path, spec: ImageSpec, arrays: dict[str, np.ndarray]):
    """Write arrays on the voxel grid as a VTK image file, for viewing.

    Parameters
    ----------
    path : pathlib.Path
        The ``.vti`` file. Its parent directory must exist.
    spec : ImageSpec
        The voxel grid.
    arrays : dict[str, numpy.ndarray]
        Point arrays by name, shaped ``(nx, ny, nz)`` for a scalar or ``(C, nx, ny, nz)`` for a vector
        with ``C`` components. Voxel ``[i, j, k]`` is at ``(x, y, z)``.
    """
    image = spec.to_vtk()
    for name, a in arrays.items():
        flat = (a.reshape(-1, order="F") if a.ndim == 3
                else np.stack([a[i].reshape(-1, order="F") for i in range(a.shape[0])], 1))
        v = numpy_to_vtk(np.ascontiguousarray(flat), deep=True)
        v.SetName(name)
        image.GetPointData().AddArray(v)
    writer = vtkXMLImageDataWriter(file_name=str(path), input_data=image)
    writer.Write()
