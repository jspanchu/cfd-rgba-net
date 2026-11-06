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
    """
    SHA-256 of a file. Cached in <file>.sha256 (with its size and mtime), since the volume files are
    tens of GB and do not change between builds.
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
    return dict(
        package_version=__version__,
        contract_version=CONTRACT_VERSION,
        label_rules_version=LABEL_RULES_VERSION,
        git=git_state(),
        inputs={k: dict(path=str(p), sha256=sha256(p)) for k, p in inputs.items() if p is not None and Path(p).exists()},
    )


CHUNK = 64     # voxels per chunk edge: a training patch reads only the chunks it overlaps
SHARD = 256    # voxels per shard edge: chunks are packed into shards, a few files per array
COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")


def _put(group, name: str, a: np.ndarray):
    """(..., nx, ny, nz) array; leading axes (channels) are kept whole in every chunk."""
    lead = a.shape[:-3]
    arr = group.create_array(name, shape=a.shape, dtype=a.dtype, chunks=lead + (CHUNK,) * 3,
                             shards=lead + (SHARD,) * 3, compressors=COMPRESSOR)
    arr[...] = a


def jsonable(obj):
    return json.loads(json.dumps(obj, default=float))


def save_case_zarr(path: Path, spec: ImageSpec, arrays: dict[str, np.ndarray],
                   groups: dict[str, dict[str, np.ndarray]], attrs: dict):
    """
    One Zarr v3 store per case: `arrays` at the root, `groups` as sub-groups of arrays, `attrs` (plus the
    grid) as the root attributes. Written to <path>.tmp and renamed at the end, so a store that exists is
    complete. An existing store is renamed aside to <path>.old first and deleted only once the new one is
    in place.
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
    image = spec.to_vtk()
    for name, a in arrays.items():
        flat = (a.reshape(-1, order="F") if a.ndim == 3
                else np.stack([a[i].reshape(-1, order="F") for i in range(a.shape[0])], 1))
        v = numpy_to_vtk(np.ascontiguousarray(flat), deep=True)
        v.SetName(name)
        image.GetPointData().AddArray(v)
    writer = vtkXMLImageDataWriter(file_name=str(path), input_data=image)
    writer.Write()

